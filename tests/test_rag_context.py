"""
Юніт-тести для rag_context.py — «колбаса»: N знайдених документів, які летять
у LLM як RAG, зібрані в ОДИН читабельний документ для показу в чаті.

МОТИВАЦІЯ. Користувач має бачити, на чому саме побудована відповідь агента, і
бачити це як суцільний текст, а не як купу JSON-об'єктів. Це вітрина контексту:
той самий набір документів, що йде в модель, але у людському вигляді.

ВАЖЛИВО (security). Цей модуль готує текст ДЛЯ ЕКРАНА. У LLM контекст іде
окремим шляхом — через prompt_guard.format_untrusted_context з делімітерами
ізоляції. Тут же головна турбота — щоб недовірений OSINT-текст не ламав
структуру самого документа (markdown-розмітка з допису).

Запуск:
    pytest tests/test_rag_context.py -v
або як звичайний скрипт (без pytest):
    python tests/test_rag_context.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_context import (  # noqa: E402
    MAX_DOC_CHARS,
    MAX_UPLOAD_CHARS,
    build_hypothesis_document,
    build_rag_document,
    decode_upload,
    hypothesis_stats,
    make_upload_doc,
    rag_stats,
)


def _doc(**kw):
    base = {
        "title": "Cisco IOS XE zero-day",
        "channel_name": "cti_channel",
        "source_type": "telegram",
        "collected_at": 1700000000,
        "language": "uk",
        "content": "Виявлено активну експлуатацію вразливості у Cisco IOS XE.",
        "summary": "Active exploitation of Cisco IOS XE.",
        "verified_cve_ids": "CVE-2023-20198",
        "verified_ttp_ids": "T1190",
        "distance": 0.0,
    }
    base.update(kw)
    return base


# --- порожній вхід -----------------------------------------------------------
def test_empty_docs_yield_empty_document():
    assert build_rag_document([]) == ""
    assert build_rag_document(None) == ""


# --- усі документи присутні --------------------------------------------------
def test_all_titles_present():
    docs = [_doc(title="Перший"), _doc(title="Другий"), _doc(title="Третій")]
    out = build_rag_document(docs)
    assert "Перший" in out and "Другий" in out and "Третій" in out


def test_documents_are_numbered():
    docs = [_doc(title="A"), _doc(title="B")]
    out = build_rag_document(docs)
    assert "1" in out and "2" in out
    # нумерація має бути саме заголовками секцій, а не випадковими цифрами
    assert out.count("###") >= 2, out


def test_full_original_text_included():
    out = build_rag_document([_doc(content="УНІКАЛЬНИЙ_ТЕКСТ_ДОКУМЕНТА")])
    assert "УНІКАЛЬНИЙ_ТЕКСТ_ДОКУМЕНТА" in out


def test_falls_back_to_summary_when_no_content():
    """content живе лише в Mongo; якщо його не дотягнули — показуємо summary,
    а не порожню секцію."""
    out = build_rag_document([_doc(content="", summary="ONLY_SUMMARY_HERE")])
    assert "ONLY_SUMMARY_HERE" in out


def test_metadata_rendered():
    out = build_rag_document([_doc(channel_name="my_channel", source_type="telegram")])
    assert "my_channel" in out
    assert "telegram" in out.lower()


def test_verified_entities_rendered():
    out = build_rag_document([_doc(verified_cve_ids="CVE-2023-20198", verified_ttp_ids="T1190")])
    assert "CVE-2023-20198" in out
    assert "T1190" in out


def test_missing_keys_do_not_crash():
    out = build_rag_document([{"title": "Голий документ"}])
    assert "Голий документ" in out


# --- обрізання довгих документів ---------------------------------------------
def test_long_document_is_trimmed_for_display():
    """Вітрина не має вивалювати мегабайт тексту в браузер — ріжемо кожен
    документ. У LLM при цьому йде ПОВНИЙ текст (інший шлях, prompt_guard)."""
    out = build_rag_document([_doc(content="я" * (MAX_DOC_CHARS * 3))])
    assert len(out) < MAX_DOC_CHARS * 3
    assert "…" in out or "..." in out


def test_trim_limit_is_configurable():
    out = build_rag_document([_doc(content="я" * 5000)], max_doc_chars=100)
    assert len(out) < 1500, len(out)


# --- недовірений markdown із допису не ламає структуру -----------------------
def test_markdown_headings_from_post_are_neutralised():
    """Допис із «### Заголовок» не має вдавати з себе секцію документа."""
    out = build_rag_document([_doc(content="### Я підроблений заголовок")])
    assert "\n### Я підроблений заголовок" not in out


def test_code_fence_from_post_cannot_break_layout():
    out = build_rag_document([_doc(content="```\nrm -rf /\n")])
    # незакрита огорожа з допису не має лишитись огорожею
    assert "\n```" not in out


def test_horizontal_rule_from_post_is_neutralised():
    out = build_rag_document([_doc(content="---")])
    assert "\n---\n" not in out


def test_post_text_is_still_readable_after_neutralising():
    """Нейтралізація не має перетворювати текст на кашу — слова лишаються."""
    out = build_rag_document([_doc(content="# Важлива новина про Cisco")])
    assert "Важлива новина про Cisco" in out


# --- статистика для шапки «колбаси» ------------------------------------------
def test_rag_stats_counts_documents_and_sources():
    docs = [_doc(channel_name="a"), _doc(channel_name="a"), _doc(channel_name="b")]
    stats = rag_stats(docs)
    assert stats["documents"] == 3
    assert stats["sources"] == 2


def test_rag_stats_counts_verified_entities_uniquely():
    docs = [
        _doc(verified_cve_ids="CVE-1, CVE-2", verified_ttp_ids="T1"),
        _doc(verified_cve_ids="CVE-2", verified_ttp_ids="T1, T2"),
    ]
    stats = rag_stats(docs)
    assert stats["cve"] == 2, stats
    assert stats["ttp"] == 2, stats


def test_rag_stats_on_empty():
    stats = rag_stats([])
    assert stats["documents"] == 0
    assert stats["characters"] == 0


def test_rag_stats_characters_counts_original_text():
    stats = rag_stats([_doc(content="12345")])
    assert stats["characters"] == 5


# --- документи, докинуті користувачем у чат ----------------------------------
def test_make_upload_doc_shape():
    """Завантажений файл стає таким самим документом контексту, як і OSINT-подія
    — щоб не плодити другу гілку обробки."""
    doc = make_upload_doc("report.txt", "Текст звіту")
    assert doc["title"] == "report.txt"
    assert doc["content"] == "Текст звіту"
    assert doc["source_type"] == "upload"
    assert doc.get("mongo_id") in (None, "")


def test_uploaded_doc_is_marked_in_document():
    """У «колбасі» має бути видно, що це файл користувача, а не джерело корпусу."""
    out = build_rag_document([make_upload_doc("report.txt", "Текст звіту")])
    assert "report.txt" in out
    assert "Текст звіту" in out


def test_upload_doc_empty_text_is_safe():
    doc = make_upload_doc("empty.txt", "")
    assert doc["content"] == ""
    assert build_rag_document([doc])  # не падає й щось таки рендерить


# --- декодування завантаженого файлу -----------------------------------------
def test_decode_upload_utf8():
    assert decode_upload("Звіт українською".encode("utf-8")) == "Звіт українською"


def test_decode_upload_accepts_str():
    """Streamlit може віддати вже декодований рядок — не падаємо."""
    assert decode_upload("вже рядок") == "вже рядок"


def test_decode_upload_broken_bytes_do_not_raise():
    """Файл у чужому кодуванні не має валити чат — псуємо символи, але живемо."""
    out = decode_upload(b"\xff\xfe\x00binary\x00")
    assert isinstance(out, str)
    assert "binary" in out


def test_decode_upload_is_truncated():
    """Дамп на десятки МБ не має ані летіти в браузер, ані з'їдати вікно моделі."""
    out = decode_upload(b"A" * (MAX_UPLOAD_CHARS * 3))
    assert len(out) == MAX_UPLOAD_CHARS


def test_decode_upload_limit_is_configurable():
    assert len(decode_upload(b"A" * 5000, max_chars=100)) == 100


def test_decode_upload_empty_and_none():
    assert decode_upload(b"") == ""
    assert decode_upload(None) == ""


# --- «колбаса» гіпотез -------------------------------------------------------
# МОТИВАЦІЯ. Гіпотези виводились як N окремих карток, у кожній ще й власний
# expander «📚 Джерела» — на 5 гіпотез це до десятка випадайок, крізь які треба
# клікати, щоб прочитати одну думку. Той самий підхід, що й для контексту RAG:
# зібрати все в ОДИН читабельний документ.
def _chain(**kw):
    base = {
        "rank": 1, "is_anchor": True,
        "nodes": ["MITRE T1566", "attacks", "ransomware"],
        "hops": 2, "total_weight": 3.671, "score": 1.835,
        "sources": [("68a1", "Фішингова розсилка з вкладенням"),
                    ("68a2", "Шифрувальник у мережі підприємства")],
    }
    base.update(kw)
    return base


def test_hypothesis_document_empty():
    assert build_hypothesis_document([], "ransomware") == ""
    assert build_hypothesis_document(None, "ransomware") == ""


def test_hypothesis_document_lists_every_chain():
    chains = [_chain(rank=1), _chain(rank=2, nodes=["CVE-2024-1", "ransomware"]),
              _chain(rank=3, nodes=["MITRE T1190", "ransomware"])]
    out = build_hypothesis_document(chains, "ransomware")
    assert "CVE-2024-1" in out and "MITRE T1190" in out and "MITRE T1566" in out
    assert out.count("###") >= 3, out


def test_hypothesis_document_renders_chain_as_arrows():
    out = build_hypothesis_document([_chain()], "ransomware")
    assert "→" in out
    assert "attacks" in out


def test_hypothesis_document_shows_metrics():
    out = build_hypothesis_document([_chain()], "ransomware")
    assert "2" in out            # кроків
    assert "3.671" in out or "3.67" in out


def test_hypothesis_document_marks_anchor_only_for_anchors():
    out = build_hypothesis_document([_chain(rank=1, is_anchor=True)], "ransomware")
    assert "якір" in out.lower(), out
    plain = build_hypothesis_document([_chain(rank=1, is_anchor=False)], "ransomware")
    assert "якір" not in plain.lower(), plain


def test_hypothesis_document_includes_sources_inline():
    """Джерела мають бути В документі, а не за ще одним кліком."""
    out = build_hypothesis_document([_chain()], "ransomware")
    assert "Фішингова розсилка з вкладенням" in out
    assert "68a1" in out


def test_hypothesis_document_without_sources_is_safe():
    out = build_hypothesis_document([_chain(sources=[])], "ransomware")
    assert "ransomware" in out


def test_hypothesis_document_missing_keys_do_not_crash():
    out = build_hypothesis_document([{"nodes": ["a", "b"]}], "b")
    assert "a" in out


# --- фіктивний початок у текстовому вигляді ----------------------------------
def test_start_label_prepended_to_every_chain():
    """Той самий «Теперішній стан», що й у графі: КОЖЕН ланцюг читається від
    єдиної точки відліку, а не починається зненацька з середини.

    Текст і граф мусять збігатись: якщо на картинці ланцюг іде «Теперішній
    стан → T1566 → …», а в тексті «T1566 → …», читач бачить дві різні
    структури й не розуміє, якій вірити."""
    chains = [_chain(rank=1), _chain(rank=2, nodes=["CVE-2024-1", "ransomware"])]
    out = build_hypothesis_document(chains, "ransomware",
                                    start_label="Теперішній стан")
    assert out.count("Теперішній стан") == len(chains), out
    assert out.index("Теперішній стан") < out.index("MITRE T1566"), out


def test_start_label_omitted_when_not_given():
    out = build_hypothesis_document([_chain()], "ransomware")
    assert "Теперішній стан" not in out


# --- недовірений текст джерел не ламає документ ------------------------------
def test_source_preview_markdown_is_neutralised():
    """Прев'ю джерела — це текст із допису (недовірений): він не має вдавати
    заголовок секції гіпотези."""
    out = build_hypothesis_document(
        [_chain(sources=[("id1", "### Підроблений заголовок")])], "ransomware")
    assert "\n### Підроблений заголовок" not in out
    assert "Підроблений заголовок" in out


# --- статистика для шапки ----------------------------------------------------
def test_hypothesis_stats_counts():
    chains = [
        _chain(rank=1, is_anchor=True, sources=[("a", "x"), ("b", "y")]),
        _chain(rank=2, is_anchor=False, sources=[("b", "y"), ("c", "z")]),
    ]
    stats = hypothesis_stats(chains)
    assert stats["hypotheses"] == 2
    assert stats["sources"] == 3        # b не рахуємо двічі
    assert stats["anchors"] == 1


def test_hypothesis_stats_empty():
    stats = hypothesis_stats([])
    assert stats["hypotheses"] == 0
    assert stats["sources"] == 0
    assert stats["anchors"] == 0


# --- ручний запуск без pytest -----------------------------------------------
if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS " + name)
            passed += 1
        except Exception:
            print("FAIL " + name)
            traceback.print_exc()
            failed += 1
    print("\n%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
