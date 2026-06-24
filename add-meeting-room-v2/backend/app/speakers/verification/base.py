from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

BackendName = Literal["speechbrain", "wespeaker"]


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    backend: BackendName
    embedding: list[float]
    load_ms: float
    inference_ms: float


class SpeakerEmbeddingBackend(Protocol):
    name: BackendName

    def embed(self, pcm: bytes, sample_rate: int) -> EmbeddingResult:
        pass


def normalize_embedding(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        raise ValueError("Speaker embedding has zero norm")
    return [value / norm for value in values]


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("No speaker embeddings to average")
    dimensions = len(embeddings[0])
    if dimensions == 0 or any(len(item) != dimensions for item in embeddings):
        raise ValueError("Speaker embedding dimensions do not match")
    averaged = [
        sum(item[index] for item in embeddings) / len(embeddings)
        for index in range(dimensions)
    ]
    return normalize_embedding(averaged)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Speaker embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))
