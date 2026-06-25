"""録音後の話者ラベル安定化テスト."""
from __future__ import annotations

import numpy as np

from das.asr.live._speaker_polish import relabel_records_by_embeddings


def test_relabel_records_by_embeddings_merges_fragmented_labels() -> None:
    records = [
        {"ms": 0, "end_ms": 2000, "speaker": "人物1", "text": "a"},
        {"ms": 2200, "end_ms": 4200, "speaker": "話者2", "text": "b"},
        {"ms": 4400, "end_ms": 6400, "speaker": "#3", "text": "c"},
        {"ms": 6600, "end_ms": 8600, "speaker": "話者2", "text": "d"},
    ]
    embeddings = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.0, 1.0]),
        2: np.array([1.0, 0.01]),
        3: np.array([0.01, 1.0]),
    }

    got = relabel_records_by_embeddings(records, embeddings, max_speakers=2)

    assert [r["speaker"] for r in got] == ["人物1", "話者2", "人物1", "話者2"]
    assert got[2]["speaker_before_polish"] == "#3"


def test_relabel_records_keeps_input_when_too_few_embeddings() -> None:
    records = [{"ms": 0, "end_ms": 2000, "speaker": "人物1", "text": "a"}]

    got = relabel_records_by_embeddings(records, {})

    assert got == records
    assert got is not records
