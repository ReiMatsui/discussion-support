"""注釈ツール（eval/annotate.py）のうち、壊れると気づきにくい所を守る.

守りたいのは2つ。

1. 自動保存が既存のGTを壊さないこと。開いただけで前の正解が消えると、
   聴き直す以外に復旧手段が無い。
2. 無音区切りが「1区間＝ひとまとまりの発話」になること。ここが崩れると
   注釈者が1区間に2人ぶんの声を聴かされ、正解そのものが濁る。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

annotate = pytest.importorskip("annotate")


def _state(tmp_path: Path):
    return annotate._State(
        name="t", title="t", wav=tmp_path / "a.wav",
        segments=[{"id": "1"}, {"id": "2"}], peaks=[], duration=1.0,
        gt_path=tmp_path / "gt_t.json", seg_path=None)


def test_save_keeps_unknown_fields_and_backs_up_once(tmp_path: Path) -> None:
    st = _state(tmp_path)
    st.gt_path.write_text(json.dumps({
        "session": "t", "transplanted_from": "別セッション",
        "speaker_names": {"S1": "話者1"}, "labels": {"1": "S1"}}),
        encoding="utf-8")

    st.save_gt({"S1": "田中", "S2": "佐藤"}, {"1": "S1", "2": "S2"})

    doc = json.loads(st.gt_path.read_text(encoding="utf-8"))
    assert doc["transplanted_from"] == "別セッション", "知らない項目が消えた"
    assert doc["labels"] == {"1": "S1", "2": "S2"}
    assert doc["speaker_names"]["S1"] == "田中"
    assert doc["labeled"] == 2
    bak = json.loads((tmp_path / "gt_t.json.bak").read_text(encoding="utf-8"))
    assert bak["labels"] == {"1": "S1"}, "上書き前の中身が退避されていない"

    st.save_gt({"S1": "田中"}, {})
    bak2 = json.loads((tmp_path / "gt_t.json.bak").read_text(encoding="utf-8"))
    assert bak2["labels"] == {"1": "S1"}, "2回目の保存で退避が上書きされた"


def test_save_drops_empty_labels(tmp_path: Path) -> None:
    st = _state(tmp_path)
    st.save_gt({}, {"1": "S1", "2": None, "3": ""})
    doc = json.loads(st.gt_path.read_text(encoding="utf-8"))
    assert doc["labels"] == {"1": "S1"}


def _tone(sec: float, hz: float = 180.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * annotate.SR)) / annotate.SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * annotate.SR), dtype=np.float32)


def test_vad_splits_on_silence_and_drops_blips() -> None:
    y = np.concatenate([
        _silence(0.5), _tone(2.0),      # 区間1
        _silence(1.0), _tone(1.5),      # 区間2
        _silence(0.8), _tone(0.1),      # 短すぎ: 落とす
        _silence(0.8), _tone(3.0),      # 区間3
    ])
    segs = annotate.segments_from_vad(y)
    assert len(segs) == 3, [(s["start"], s["end"]) for s in segs]
    assert [s["id"] for s in segs] == ["1", "2", "3"]
    assert segs[0]["start"] == pytest.approx(0.5, abs=0.1)
    assert segs[0]["end"] == pytest.approx(2.5, abs=0.15)
    assert all(s["end"] > s["start"] for s in segs)


def test_vad_keeps_short_pauses_inside_one_utterance() -> None:
    """息継ぎ（0.2秒）で切らない——1区間が細切れだと注釈が終わらない."""
    y = np.concatenate([_silence(0.3), _tone(1.2), _silence(0.2), _tone(1.2)])
    segs = annotate.segments_from_vad(y)
    assert len(segs) == 1


def test_vad_splits_overlong_span() -> None:
    """長すぎる区間は割る（1区間に複数話者が入る確率を下げるため）."""
    y = np.concatenate([_silence(0.3), _tone(30.0)])
    segs = annotate.segments_from_vad(y)
    assert len(segs) >= 3
    assert max(s["end"] - s["start"] for s in segs) <= 12.5
    # 割っても時間の連続性は保つ（聴き落としが出ないように）
    for a, b in itertools.pairwise(segs):
        assert b["start"] == pytest.approx(a["end"], abs=0.05)
