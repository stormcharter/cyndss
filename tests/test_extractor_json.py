"""
Терпимий розбір JSON-відповіді екстрактора + response_format у запиті.

Дефект (виміряно наживо, Ollama hermes3:8b): 9 зі 53 статей отримали
"Unparsed Threat Event" і порожній summary — модель закінчувала відповідь без
закриваючої `}` КОРЕНЕВОГО об'єкта, а зріз `find("{") : rfind("}")+1` ловив `}`
вкладеного об'єкта й давав JSONDecodeError (finish_reason=stop, не обрізка).

Запуск:
    python tests/test_extractor_json.py
або pytest.
"""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Settings читає .env з cwd; легасі-ключі в локальному .env ламають імпорт
# (див. tests/test_verifier_revoked.py) — на час імпорту йдемо в порожній каталог.
_cwd_before_import = os.getcwd()
with tempfile.TemporaryDirectory() as _tmpdir:
    os.chdir(_tmpdir)
    try:
        import httpx  # noqa: E402
        from shipper.extractor import OllamaExtractor, parse_model_json  # noqa: E402
    finally:
        os.chdir(_cwd_before_import)


# Реальна форма збою: масив об'єктів із quote, далі detected_language, і НЕМАЄ
# фінальної `}` кореня.
REAL_BROKEN = """{
  "title": "RDP brute force campaign",
  "summary": "Attackers used RDP for initial access.",
  "extracted_cve_ids": [],
  "mitre_tactics_techniques": [
    {
      "id": "T1021.001",
      "quote": "...via RDP."
    }
  ],
  "detected_language": "en"
"""


# --- parse_model_json -------------------------------------------------------
def test_valid_json_unchanged():
    assert parse_model_json('{"a": 1, "b": [1, 2]}') == {"a": 1, "b": [1, 2]}


def test_json_in_wrapper_text():
    assert parse_model_json('Sure: {"a": 1} done') == {"a": 1}


def test_missing_root_brace_real_example():
    got = parse_model_json(REAL_BROKEN)
    assert got is not None
    assert got["title"] == "RDP brute force campaign"
    assert got["mitre_tactics_techniques"] == [{"id": "T1021.001", "quote": "...via RDP."}]
    assert got["detected_language"] == "en"


def test_missing_bracket_and_brace_closed_in_right_order():
    got = parse_model_json('{"a": [{"id": "x"}, {"id": "y"}')
    assert got == {"a": [{"id": "x"}, {"id": "y"}]}


def test_missing_closers_inside_nested_object():
    assert parse_model_json('{"a": {"b": [1, 2') == {"a": {"b": [1, 2]}}


def test_trailing_comma_before_end():
    assert parse_model_json('{"a": [1, 2],') == {"a": [1, 2]}
    assert parse_model_json('{"a": [1, 2,\n  ') == {"a": [1, 2]}


def test_brackets_and_quotes_inside_strings_do_not_break_counting():
    raw = '{"quote": "use {x} and [y] \\"q\\"", "n": [1'
    assert parse_model_json(raw) == {"quote": 'use {x} and [y] "q"', "n": [1]}


def test_backslash_before_closing_quote_in_string():
    # рядок закінчується екранованим слешем: "c:\\" — лапка після нього РЕАЛЬНО закриває рядок
    assert parse_model_json('{"p": "c:\\\\", "n": [1') == {"p": "c:\\", "n": [1]}


def test_garbage_without_brace_is_none():
    assert parse_model_json("no json here at all") is None
    assert parse_model_json("") is None


def test_truncated_inside_string_is_none():
    # Обірваний посеред рядка вивід НЕ латаємо: закрита лапка дала б
    # відрізаний title/quote, тобто дані, яких модель не віддавала.
    assert parse_model_json('{"title": "RDP brute for') is None


def test_repair_does_not_invent_values():
    # висячий ключ без значення — ремонт не вигадує значення
    assert parse_model_json('{"a": 1, "b":') is None


def test_non_dict_root_is_none():
    assert parse_model_json("[1, 2, 3]") is None


# --- запит і extract --------------------------------------------------------
def _extractor(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaExtractor(client=client)


def _chat_reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_payload_requests_json_object():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _chat_reply("{}")

    asyncio.run(_extractor(handler).extract("text"))
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_extract_survives_missing_root_brace():
    ext = _extractor(lambda request: _chat_reply(REAL_BROKEN))
    res = asyncio.run(ext.extract("Attackers moved via RDP."))
    assert res.title == "RDP brute force campaign"
    assert res.summary == "Attackers used RDP for initial access."
    assert res.mitre_tactics_techniques == ["T1021.001"]
    assert res.ttp_contexts == {"T1021.001": "...via RDP."}
    assert res.detected_language == "en"


def test_extract_unparseable_still_degrades_to_empty():
    ext = _extractor(lambda request: _chat_reply("sorry, no json"))
    res = asyncio.run(ext.extract("some text"))
    assert res.title == "Unparsed Threat Event"
    assert res.summary == "some text"


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
