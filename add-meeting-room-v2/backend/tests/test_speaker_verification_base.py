import pytest

from app.speakers.verification.base import (
    average_embeddings,
    cosine_similarity,
    normalize_embedding,
)


def test_normalize_and_cosine_similarity() -> None:
    left = normalize_embedding([3.0, 4.0])
    right = normalize_embedding([6.0, 8.0])

    assert left == pytest.approx([0.6, 0.8])
    assert cosine_similarity(left, right) == pytest.approx(1.0)


def test_average_embeddings_normalizes_result() -> None:
    averaged = average_embeddings([[1.0, 0.0], [0.0, 1.0]])

    assert averaged == pytest.approx([2**-0.5, 2**-0.5])
