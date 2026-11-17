"""
Юніт-тести tools/reembed_index.py — міграції векторів індексу на нову схему
ембедингів («title. summary» замість сирого оригіналу).

КОНТЕКСТ. Індекс (1009 документів) має вектори зі СИРОГО українського тексту;
шиппер тепер рахує їх з англійського «title. summary» (embedding_text.py).
Змішувати схеми не можна, тож інструмент перераховує вектор КОЖНОГО документа й
лише тоді ставить маркер, за яким шиппер дозволяє собі писати в індекс.

Що перевіряється без мережі й баз: вибір тексту, нормалізація, батчинг із
відкатом на поштучне ембедування, збір документа під REPLACE (жодного атрибута
не загублено; content лематизований, як у шиппера), рішення про оренду шиппера,
рішення про маркер і ПОВНИЙ потік `execute` на фейковому I/O: dry-run нічого не
пише, apply не пише маркер при жодному збої.

Запуск:
    python tests/test_reembed_index.py
або pytest.
"""
import math
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_text import EMBED_TEXT_CHARS, SUMMARY_PLACEHOLDER  # noqa: E402
from lemmatizer import lemmatize_text  # noqa: E402
from shipper.embedding_scheme import MARKER_ID, scheme_for  # noqa: E402
from tools.purge_ungrounded_ttps import build_replace_doc as purge_replace_doc  # noqa: E402
from tools.reembed_index import (  # noqa: E402
    LEASE_DOC_ID,
    SCAN_COLS,
    STATE_COLLECTION,
    build_parser,
    build_reembedded_doc,
    chunked,
    embed_in_batches,
    embed_text_for,
    execute,
    lease_block_reason,
    lease_is_active,
    normalize_vector,
    plan_items,
    shipper_view,
    should_write_marker,
    valid_vector,
)

NOW = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)

ROW = {
    "id": 42,
    "mongo_id": "507f1f77bcf86cd799439011",
    "source_type": "telegram",
    "channel_name": "chan",
    "url": "https://t.me/chan/1",
    "collected_at": 1780000000,
    "title": "Mail gateway attacked",
    "language": "uk",
    "summary": "Attackers hit a mail gateway.",
    "verified_cve_ids": "CVE-2026-3055",
    "verified_ttp_ids": "T1566",
    "cve_count_verified": 1,
    "ttp_count_verified": 1,
    "ioc_count": 3,
}
TEXT = "Дослідники повідомили про атаку на поштовий шлюз."


def vec(seed: float = 1.0, dims: int = 768) -> list[float]:
    return [seed + i * 0.001 for i in range(dims)]


# --- вибір тексту -----------------------------------------------------------
def test_embed_text_is_title_dot_summary_from_the_index_row():
    assert embed_text_for(ROW, TEXT) == "Mail gateway attacked. Attackers hit a mail gateway."


def test_embed_text_falls_back_to_the_mongo_text_for_placeholder_rows():
    row = dict(ROW, title="Untitled", summary=SUMMARY_PLACEHOLDER)
    assert embed_text_for(row, TEXT) == TEXT


def test_embed_text_is_capped():
    row = dict(ROW, title="", summary="")
    assert len(embed_text_for(row, "к" * 5000)) == EMBED_TEXT_CHARS


# --- вектор -----------------------------------------------------------------
def test_normalize_vector_gives_unit_length_like_the_shipper():
    out = normalize_vector([3.0, 4.0])
    assert out == [0.6, 0.8]
    assert math.isclose(math.sqrt(sum(x * x for x in normalize_vector(vec()))), 1.0)


def test_normalize_vector_leaves_zero_vector_alone():
    # Як у EmbeddingClient: `if norm > 0`. Валідність відсіє valid_vector.
    assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]


def test_valid_vector_requires_768_finite_numbers():
    assert valid_vector(vec())
    assert not valid_vector(vec(dims=767))
    assert not valid_vector([])
    assert not valid_vector(vec()[:-1] + [float("nan")])
    assert not valid_vector(vec()[:-1] + [float("inf")])


# --- батчинг ----------------------------------------------------------------
def test_chunked_splits_preserving_order():
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunked([], 3) == []


class BatchEmbed:
    """Фейковий ембедер: падає на пакетах, що містять «BAD»; пам'ятає виклики."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        if any("BAD" in t for t in texts):
            raise RuntimeError("HTTP 400: input length exceeds the context length")
        return [[float(len(t))] for t in texts]


def test_embed_in_batches_preserves_order_and_batch_size():
    embed = BatchEmbed()
    out = embed_in_batches(embed, ["a", "bb", "ccc", "dddd", "eeeee"], 2)
    assert out == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert [len(c) for c in embed.calls] == [2, 2, 1]


def test_embed_in_batches_isolates_a_bad_text_by_falling_back_one_by_one():
    embed = BatchEmbed()
    out = embed_in_batches(embed, ["a", "BAD", "ccc"], 3)
    assert out == [[1.0], None, [3.0]]


def test_embed_in_batches_treats_a_wrong_count_as_a_batch_failure():
    def short(texts):
        return [[1.0]] if len(texts) > 1 else [[7.0]]

    assert embed_in_batches(short, ["a", "b"], 2) == [[7.0], [7.0]]


# --- документ під REPLACE ---------------------------------------------------
def test_reembedded_doc_carries_exactly_the_attributes_of_the_purge_tool():
    mine = build_reembedded_doc(ROW, TEXT, vec())
    theirs = purge_replace_doc(ROW, TEXT, vec(), ["CVE-2026-3055"], ["T1566"])
    assert set(mine) == set(theirs)
    assert "id" not in mine


def test_reembedded_doc_keeps_every_stored_field_and_swaps_only_the_vector():
    doc = build_reembedded_doc(ROW, TEXT, vec(2.0))
    for key in ("mongo_id", "source_type", "channel_name", "url", "collected_at",
                "title", "language", "summary", "verified_cve_ids", "verified_ttp_ids",
                "cve_count_verified", "ttp_count_verified", "ioc_count"):
        assert doc[key] == ROW[key], key
    assert doc["embedding"] == vec(2.0)


def test_reembedded_doc_content_is_lemmatized_like_the_shipper_writes_it():
    # manticore_sink._to_manticore_doc кладе lemmatize_text(doc.text); REPLACE із
    # сирим текстом тихо зіпсував би пошук по content.
    assert build_reembedded_doc(ROW, TEXT, vec())["content"] == lemmatize_text(TEXT)
    assert build_reembedded_doc(ROW, "plain english", vec())["content"] == "plain english"


def test_reembedded_doc_defaults_missing_fields_like_the_purge_tool():
    doc = build_reembedded_doc({"mongo_id": "m"}, "x", vec())
    assert doc["channel_name"] == "" and doc["collected_at"] == 0 and doc["ioc_count"] == 0


# --- план -------------------------------------------------------------------
def test_plan_items_splits_readable_rows_from_unreadable():
    rows = [ROW, dict(ROW, mongo_id="a" * 24), dict(ROW, mongo_id="b" * 24)]
    texts = {ROW["mongo_id"]: TEXT, "b" * 24: "   "}
    items, skipped = plan_items(rows, texts)
    assert [r["mongo_id"] for r, _t, _e in items] == [ROW["mongo_id"]]
    assert items[0][1] == TEXT
    assert items[0][2] == embed_text_for(ROW, TEXT)
    assert [r["mongo_id"] for r, _reason in skipped] == ["a" * 24, "b" * 24]


def test_plan_items_skips_rows_with_nothing_to_embed():
    row = dict(ROW, title="Untitled", summary=SUMMARY_PLACEHOLDER)
    items, skipped = plan_items([row], {row["mongo_id"]: "   "})
    assert items == [] and len(skipped) == 1


# --- оренда шиппера ---------------------------------------------------------
def test_lease_is_active_only_with_owner_and_future_expiry():
    live = {"_id": LEASE_DOC_ID, "lease_owner": "w1", "lease_expires_at": NOW + timedelta(seconds=30)}
    assert lease_is_active(live, NOW)
    assert not lease_is_active(None, NOW)
    assert not lease_is_active({"_id": LEASE_DOC_ID}, NOW)
    assert not lease_is_active(dict(live, lease_owner=None), NOW)
    assert not lease_is_active(dict(live, lease_owner=""), NOW)
    assert not lease_is_active(dict(live, lease_expires_at=NOW - timedelta(seconds=1)), NOW)
    assert not lease_is_active(dict(live, lease_expires_at=NOW), NOW)  # не в майбутньому
    assert not lease_is_active(dict(live, lease_expires_at=None), NOW)


def test_lease_with_naive_datetime_is_read_as_utc():
    # pymongo без tz_aware віддає naive UTC.
    live = {"lease_owner": "w1", "lease_expires_at": (NOW + timedelta(seconds=30)).replace(tzinfo=None)}
    assert lease_is_active(live, NOW)


def test_lease_block_reason_refuses_while_shipper_is_alive_unless_forced():
    live = {"lease_owner": "w1", "lease_expires_at": NOW + timedelta(seconds=30)}
    reason = lease_block_reason(live, False, NOW)
    assert reason and "шиппер" in reason.lower() and "--force" in reason
    assert lease_block_reason(live, True, NOW) is None
    assert lease_block_reason(None, False, NOW) is None


def test_state_constants_match_the_shipper_ones():
    from shipper.mongo_source import STATE_COLLECTION as SHIPPER_STATE, STATE_DOC_ID

    assert STATE_COLLECTION == SHIPPER_STATE
    assert LEASE_DOC_ID == STATE_DOC_ID


# --- маркер -----------------------------------------------------------------
def test_orphans_block_the_marker_until_they_are_dropped():
    # Осиротілі рядки лишаються в індексі зі СТАРИМИ векторами: поки вони там,
    # маркер ставити не можна. Видалені (dropped) — вже не рахуються, а
    # count(*) після прогону має дорівнювати scanned - dropped.
    assert not should_write_marker(failed=0, scanned=10, in_index=10, orphans=2)
    assert should_write_marker(failed=0, scanned=10, in_index=8, orphans=0, dropped=2)
    assert not should_write_marker(failed=0, scanned=10, in_index=10, orphans=0, dropped=2)
    assert not should_write_marker(failed=1, scanned=10, in_index=8, orphans=0, dropped=2)
    # індекс спорожнів повністю — маркер не потрібен (шиппер сам поставить)
    assert not should_write_marker(failed=0, scanned=2, in_index=0, orphans=0, dropped=2)


def test_marker_only_after_a_fully_successful_run():
    assert should_write_marker(failed=0, scanned=1009, in_index=1009)
    assert not should_write_marker(failed=1, scanned=1009, in_index=1009)
    # REPLACE не має розмножувати/губити документи: кількість мусить зійтись.
    assert not should_write_marker(failed=0, scanned=1009, in_index=1010)
    assert not should_write_marker(failed=0, scanned=1009, in_index=1008)
    assert not should_write_marker(failed=0, scanned=0, in_index=0)


# --- CLI --------------------------------------------------------------------
def test_cli_defaults_are_a_dry_run_against_local_ollama():
    args = build_parser().parse_args([])
    assert args.apply is False and args.force is False
    assert args.ollama_url == "http://127.0.0.1:11434"
    assert args.embed_model == "nomic-embed-text"
    assert args.batch == 32
    assert args.manticore_url == "http://127.0.0.1:9998"
    assert args.table == "events_enriched"


def test_cli_scheme_matches_the_shipper_default():
    from shipper.config import Settings
    from shipper.embedding_scheme import current_scheme

    args = build_parser().parse_args([])
    assert scheme_for(args.embed_model) == current_scheme(Settings(_env_file=None))


def test_embed_model_defaults_to_the_shippers_ollama_embed_model_from_env():
    # Маркер пишеться за ім'ям моделі; якщо .env шиппера каже «:latest», а
    # інструмент мовчки бере «nomic-embed-text», маркер не збігається ніколи.
    args = build_parser(env={"OLLAMA_EMBED_MODEL": "nomic-embed-text:latest"}).parse_args([])
    assert args.embed_model == "nomic-embed-text:latest"
    assert build_parser(env={}).parse_args([]).embed_model == "nomic-embed-text"
    explicit = build_parser(env={"OLLAMA_EMBED_MODEL": "x"}).parse_args(["--embed-model", "y"])
    assert explicit.embed_model == "y"


def test_shipper_view_reads_provider_and_models_from_env():
    view = shipper_view(env={"EMBED_PROVIDER": " Remote ", "REMOTE_EMBED_MODEL": "R",
                             "OLLAMA_EMBED_MODEL": "O"})
    assert (view.embed_provider, view.remote_embed_model, view.ollama_embed_model) == \
        ("remote", "R", "O")
    default = shipper_view(env={})
    assert default.embed_provider == "ollama" and default.ollama_embed_model == "nomic-embed-text"


def test_tool_and_shipper_agree_on_the_scheme_for_ollama():
    # Інструмент рахує очікувану схему шиппера з env за тим самим правилом, що
    # й current_scheme(Settings): для ollama обидва мусять дати той самий рядок.
    from shipper.config import Settings
    from shipper.embedding_scheme import current_scheme

    for model in ("nomic-embed-text", "nomic-embed-text:latest"):
        env = {"EMBED_PROVIDER": "ollama", "OLLAMA_EMBED_MODEL": model}
        cfg = Settings(_env_file=None, embed_provider="ollama", ollama_embed_model=model)
        assert current_scheme(shipper_view(env=env)) == current_scheme(cfg)
        assert scheme_for(build_parser(env=env).parse_args([]).embed_model) == current_scheme(cfg)


def test_scan_columns_do_not_pull_the_old_vectors():
    # Старий вектор не потрібен: ми його замінюємо; менше трафіку з Manticore.
    assert "embedding" not in SCAN_COLS
    assert "mongo_id" in SCAN_COLS and "summary" in SCAN_COLS


# --- повний потік на фейковому I/O ------------------------------------------
class FakeIO:
    def __init__(self, rows, texts, lease=None, bad_texts=(), in_index=None, replace_errors=(),
                 replace_raises=(), delete_fails=False):
        self.rows = rows
        self.texts = texts
        self.lease = lease
        self.bad_texts = set(bad_texts)
        self.in_index = len(rows) if in_index is None else in_index
        self.replace_errors = set(replace_errors)
        self.replace_raises = set(replace_raises)
        self.delete_fails = delete_fails
        self.deleted: list[list[int]] = []
        self.embedded: list[str] = []
        self.replaced: list[tuple[int, dict]] = []
        self.markers: list[tuple[str, int]] = []

    def scan_rows(self):
        return list(self.rows)

    def fetch_texts(self, mongo_ids):
        return {m: self.texts[m] for m in mongo_ids if m in self.texts}

    def lease_state(self):
        return self.lease

    def embed(self, texts):
        self.embedded.extend(texts)
        if self.bad_texts & set(texts):
            raise RuntimeError("HTTP 400")
        return [vec(1.0 + len(t)) for t in texts]

    def replace(self, doc_id, doc):
        self.replaced.append((doc_id, doc))
        if doc["mongo_id"] in self.replace_raises:
            raise OSError("urlopen error: timed out")   # як HTTPError/URLError з bulk_replace
        return doc["mongo_id"] not in self.replace_errors

    def delete(self, ids):
        self.deleted.append(list(ids))
        if self.delete_fails:
            raise OSError("Manticore SQL error")
        self.in_index -= len(ids)

    def count(self):
        return self.in_index

    def write_marker(self, scheme, documents):
        self.markers.append((scheme, documents))


def _rows(n):
    return [dict(ROW, id=i + 1, mongo_id=f"{i + 1:024x}", title=f"T{i}", summary=f"S{i}")
            for i in range(n)]


def _texts(rows):
    return {r["mongo_id"]: TEXT for r in rows}


def _args(*extra):
    return build_parser(env={}).parse_args(list(extra))   # env={}: не залежимо від локального .env


OLLAMA_SHIPPER = shipper_view(env={})   # шиппер за замовчуванням: ollama / nomic-embed-text


def _run(io, *flags, shipper=OLLAMA_SHIPPER, args=None):
    lines = []
    code = execute(args or _args(*flags), io, out=lines.append, shipper=shipper)
    return code, "\n".join(lines)


def test_dry_run_writes_nothing_and_says_so():
    rows = _rows(3)
    io = FakeIO(rows, _texts(rows))
    code, text = _run(io)
    assert code == 0
    assert io.replaced == [] and io.markers == []
    assert "DRY-RUN" in text
    assert len(io.embedded) <= 1  # лише проба зв'язку з Ollama


def test_apply_reembeds_every_document_and_then_writes_the_marker():
    rows = _rows(3)
    io = FakeIO(rows, _texts(rows))
    code, text = _run(io, "--apply")
    assert code == 0
    assert [d for d, _doc in io.replaced] == [1, 2, 3]
    assert io.embedded == ["T0. S0", "T1. S1", "T2. S2"]
    for _id, doc in io.replaced:
        assert len(doc["embedding"]) == 768
        assert math.isclose(math.sqrt(sum(x * x for x in doc["embedding"])), 1.0)
    assert io.markers == [(scheme_for("nomic-embed-text"), 3)]
    assert "3" in text


def test_apply_refuses_while_the_shipper_holds_the_lease():
    rows = _rows(2)
    live = {"lease_owner": "w1", "lease_expires_at": datetime.now(timezone.utc) + timedelta(seconds=30)}
    io = FakeIO(rows, _texts(rows), lease=live)
    code, text = _run(io, "--apply")
    assert code == 3
    assert io.replaced == [] and io.markers == [] and io.embedded == []
    assert "--force" in text


def test_apply_with_force_ignores_the_lease():
    rows = _rows(2)
    live = {"lease_owner": "w1", "lease_expires_at": datetime.now(timezone.utc) + timedelta(seconds=30)}
    io = FakeIO(rows, _texts(rows), lease=live)
    code, _text = _run(io, "--apply", "--force")
    assert code == 0 and len(io.replaced) == 2


def test_dry_run_does_not_care_about_the_lease():
    rows = _rows(2)
    live = {"lease_owner": "w1", "lease_expires_at": datetime.now(timezone.utc) + timedelta(seconds=30)}
    assert _run(FakeIO(rows, _texts(rows), lease=live))[0] == 0


def test_a_failed_embedding_means_no_marker_and_a_nonzero_exit():
    rows = _rows(3)
    io = FakeIO(rows, _texts(rows), bad_texts={"T1. S1"})
    code, text = _run(io, "--apply", "--batch", "3")
    assert code == 1
    assert io.markers == []
    assert [d for d, _doc in io.replaced] == [1, 3]  # решта все одно переїхала
    assert "failed" in text.lower() or "помил" in text.lower()


def test_a_replace_error_means_no_marker():
    rows = _rows(2)
    io = FakeIO(rows, _texts(rows), replace_errors={rows[0]["mongo_id"]})
    code, _text = _run(io, "--apply")
    assert code == 1 and io.markers == []


def test_a_row_without_source_text_blocks_the_marker():
    rows = _rows(2)
    texts = _texts(rows)
    del texts[rows[1]["mongo_id"]]
    io = FakeIO(rows, texts)
    code, _text = _run(io, "--apply")
    assert code == 1 and io.markers == []
    assert [d for d, _doc in io.replaced] == [1]  # без тексту REPLACE затер би content


def test_a_replace_exception_does_not_abort_the_pass():
    # bulk_replace -> urlopen кидає HTTPError/URLError/timeout: це збій ОДНОГО
    # документа, а не всього проходу. Решта все одно переїжджає, звіт друкується,
    # код виходу 1, маркера немає.
    rows = _rows(5)
    io = FakeIO(rows, _texts(rows), replace_raises={rows[1]["mongo_id"]})
    code, text = _run(io, "--apply")
    assert code == 1 and io.markers == []
    assert [d for d, _doc in io.replaced] == [1, 2, 3, 4, 5]   # 3..5 оброблено
    assert "failed=1" in text and "re-embedded=4" in text
    assert "timed out" in text


def test_document_count_drift_blocks_the_marker():
    rows = _rows(2)
    io = FakeIO(rows, _texts(rows), in_index=3)
    code, _text = _run(io, "--apply")
    assert code == 1 and io.markers == []


# --- осиротілі рядки --------------------------------------------------------
def _with_orphans(n_good, n_orphans):
    rows = _rows(n_good + n_orphans)
    texts = _texts(rows)
    orphans = rows[n_good:]
    for r in orphans:
        del texts[r["mongo_id"]]
    return rows, texts, orphans


def test_orphans_are_counted_separately_from_failures_and_block_the_marker():
    rows, texts, orphans = _with_orphans(2, 1)
    io = FakeIO(rows, texts)
    code, text = _run(io, "--apply")
    assert code == 1 and io.markers == [] and io.deleted == []
    assert "orphans=1" in text and "failed=0" in text
    assert "--apply --drop-orphans" in text   # що саме повторити


def test_the_full_orphan_list_is_printed_not_just_the_first_twenty():
    rows, texts, orphans = _with_orphans(1, 25)
    code, text = _run(FakeIO(rows, texts), "--apply")
    assert code == 1
    for r in orphans:
        assert r["mongo_id"] in text, r["mongo_id"]


def test_dry_run_lists_orphans_and_mentions_drop_orphans():
    rows, texts, orphans = _with_orphans(1, 22)
    io = FakeIO(rows, texts)
    code, text = _run(io)
    assert code == 0 and io.deleted == [] and io.replaced == []
    assert all(r["mongo_id"] in text for r in orphans)
    assert "--drop-orphans" in text


def test_drop_orphans_without_apply_deletes_nothing():
    rows, texts, _orphans = _with_orphans(1, 2)
    io = FakeIO(rows, texts)
    code, _text = _run(io, "--drop-orphans")
    assert code == 0 and io.deleted == [] and io.replaced == [] and io.markers == []


def test_apply_with_drop_orphans_deletes_them_and_then_writes_the_marker():
    rows, texts, orphans = _with_orphans(2, 2)
    io = FakeIO(rows, texts)
    code, text = _run(io, "--apply", "--drop-orphans")
    assert code == 0
    assert [i for chunk in io.deleted for i in chunk] == [r["id"] for r in orphans]
    assert [d for d, _doc in io.replaced] == [1, 2]      # осиротілі не ембедяться
    assert io.markers == [(scheme_for("nomic-embed-text"), 2)]   # count(*) = 2 після видалення
    assert "dropped=2" in text


def test_drop_orphans_deletes_in_chunks():
    rows, texts, orphans = _with_orphans(1, 1200)
    io = FakeIO(rows, texts)
    code, _text = _run(io, "--apply", "--drop-orphans")
    assert code == 0 and io.markers == [(scheme_for("nomic-embed-text"), 1)]
    assert [len(c) for c in io.deleted] == [500, 500, 200]


def test_a_failed_delete_counts_as_failed_and_blocks_the_marker():
    rows, texts, _orphans = _with_orphans(2, 2)
    io = FakeIO(rows, texts, delete_fails=True)
    code, text = _run(io, "--apply", "--drop-orphans")
    assert code == 1 and io.markers == []
    assert "failed=2" in text            # два рядки з невдалого пакета
    assert [d for d, _doc in io.replaced] == [1, 2]


def test_orphan_with_an_invalid_mongo_id_is_an_orphan_too():
    # fetch_texts відсіює не-ObjectId ДО запиту, тож текст для такого рядка
    # не знайдеться ніколи: це той самий випадок, що й відсутній текст.
    rows = _rows(2)
    rows[1]["mongo_id"] = "legacy-not-an-objectid"
    texts = {rows[0]["mongo_id"]: TEXT}
    io = FakeIO(rows, texts)
    code, text = _run(io, "--apply", "--drop-orphans")
    assert code == 0 and io.deleted == [[2]]
    assert io.markers == [(scheme_for("nomic-embed-text"), 1)]
    assert "legacy-not-an-objectid" in text


def test_dropping_every_row_leaves_an_empty_index_without_a_marker():
    rows, texts, _orphans = _with_orphans(0, 2)
    io = FakeIO(rows, texts)
    code, _text = _run(io, "--apply", "--drop-orphans")
    assert io.markers == []      # порожній індекс шиппер позначить сам
    assert code == 0


# --- схема: що запишемо й що очікує шиппер -----------------------------------
def test_both_schemes_are_printed_in_dry_run_and_apply():
    rows = _rows(2)
    expected = scheme_for("nomic-embed-text")
    for flags in ((), ("--apply",)):
        _code, text = _run(FakeIO(rows, _texts(rows)), *flags)
        assert "схема, яку запише інструмент" in text and "схема, яку очікує шиппер" in text
        assert text.count(expected) >= 2


def test_a_custom_shipper_model_that_the_tool_does_not_use_is_flagged_as_mismatch():
    rows = _rows(2)
    shipper = shipper_view(env={"OLLAMA_EMBED_MODEL": "nomic-embed-text:latest"})
    _code, text = _run(FakeIO(rows, _texts(rows)), "--apply", shipper=shipper)
    assert "не збігається" in text
    assert scheme_for("nomic-embed-text:latest") in text


def test_matching_custom_model_gives_no_mismatch_warning():
    rows = _rows(2)
    env = {"OLLAMA_EMBED_MODEL": "nomic-embed-text:latest"}
    io = FakeIO(rows, _texts(rows))
    args = build_parser(env=env).parse_args(["--apply"])
    code, text = _run(io, args=args, shipper=shipper_view(env=env))
    assert code == 0 and "не збігається" not in text
    assert io.markers == [(scheme_for("nomic-embed-text:latest"), 2)]


def test_apply_is_refused_when_the_shipper_embeds_remotely_or_auto():
    rows = _rows(2)
    for provider in ("remote", "auto"):
        shipper = shipper_view(env={"EMBED_PROVIDER": provider})
        io = FakeIO(rows, _texts(rows))
        code, text = _run(io, "--apply", shipper=shipper)
        assert code == 4, provider
        assert io.embedded == [] and io.replaced == [] and io.markers == []
        assert provider in text and "Ollama" in text


def test_dry_run_with_a_remote_shipper_warns_but_does_not_fail():
    rows = _rows(2)
    code, text = _run(FakeIO(rows, _texts(rows)), shipper=shipper_view(env={"EMBED_PROVIDER": "auto"}))
    assert code == 0 and "--apply" in text and "auto" in text


# --- LiveIO.delete (мережа підмінена) ----------------------------------------
def test_live_io_delete_sends_a_delete_by_id_query():
    import tools.reembed_index as mod

    sent = []
    original = mod.mano_sql
    mod.mano_sql = lambda url, query: sent.append((url, query)) or {}   # без pytest-monkeypatch
    try:
        io = mod.LiveIO(_args("--table", "events_x", "--manticore-url", "http://m:1"), "mongodb://x")
        io.delete([4, -1])
    finally:
        mod.mano_sql = original
    assert sent == [("http://m:1", f"DELETE FROM events_x WHERE id IN (4,{2**64 - 1})")]


def test_apply_is_idempotent():
    rows = _rows(2)
    first, second = FakeIO(rows, _texts(rows)), FakeIO(rows, _texts(rows))
    _run(first, "--apply")
    _run(second, "--apply")
    assert first.replaced == second.replaced
    assert first.markers == second.markers


def test_empty_index_is_not_marked():
    code, _text = _run(FakeIO([], {}), "--apply")
    assert code == 0  # нічого переробляти
    # порожній індекс шиппер сам позначить на старті (ensure_index_scheme)


def test_marker_id_is_the_agreed_one():
    assert MARKER_ID == "index_embedding"


# --- ручний запуск без pytest --------------------------------------------
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
