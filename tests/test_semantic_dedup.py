"""Тести порога та поведінки вікна SemanticDeduplicator."""

import numpy as np
import pytest

from shipper.dedup import SemanticDeduplicator


def _normalized(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    return arr / n if n > 0 else arr


@pytest.fixture
def dedup() -> SemanticDeduplicator:
    return SemanticDeduplicator()


def test_first_candidate_is_not_remembered_before_indexing(
    dedup: SemanticDeduplicator,
) -> None:
    embedding = _normalized([1.0, 0.0, 0.0])

    result = dedup.find_duplicate(embedding)

    assert not result.is_duplicate
    assert len(dedup.window) == 0


def test_identical_text_detected_after_canonical_is_indexed(
    dedup: SemanticDeduplicator,
) -> None:
    embedding = _normalized([1.0, 0.0, 0.0])
    dedup.remember_indexed("id1", "original text", embedding)

    result = dedup.find_duplicate(embedding)

    assert result.is_duplicate
    assert result.duplicate_of_mongo_id == "id1"
    assert result.similarity >= 0.99


def test_unrelated_text_not_duplicate(dedup: SemanticDeduplicator) -> None:
    dedup.remember_indexed(
        "id1",
        "original text",
        _normalized([1.0, 0.0, 0.0]),
    )

    result = dedup.find_duplicate(_normalized([0.0, 1.0, 0.0]))

    assert not result.is_duplicate
    assert result.similarity < 0.3


def test_near_threshold_edge(dedup: SemanticDeduplicator) -> None:
    canonical = _normalized([1.0, 0.0, 0.0])
    candidate = _normalized([0.8, 0.6, 0.0])
    dedup.remember_indexed("id1", "a", canonical)

    result = dedup.find_duplicate(candidate)

    assert result.similarity < 0.92
    assert not result.is_duplicate


def test_window_maxlen() -> None:
    small = SemanticDeduplicator()
    small.window = type(small.window)(maxlen=3)
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.707, 0.707, 0.0],
        [0.707, 0.0, 0.707],
        [0.0, 0.707, 0.707],
    ]
    for index in range(10):
        embedding = _normalized(vectors[index % len(vectors)])
        small.remember_indexed(f"id{index}", f"text{index}", embedding)

    assert len(small.window) == 3
    assert small.window[0].mongo_id == "id7"
