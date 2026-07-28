"""声紋判定（`VoiceProfiles._classify`）の出力を丸ごと固定する検証台.

`flush` と同じ狙い（`test_flush_golden.py`）。ここは帰属の一段目で、返した
キーだけでなく **kind（判定の種別）と、判定が書き換えた台帳** が下流の全部を
動かす——`kind` は席の割当ての条件になり、`sp_map` は次の発話のラベル継続を
決め、`profiles` は以後の照合相手になる。返り値だけ見ていても壊れたことに
気づけない。

台本は判定の主要な経路を通す:

  1. 未知の声（照合相手なし）→ 蓄積が始まる
  2. 同じ声が続いて自動登録に達する
  3. 登録済みの声が再訪 → 声紋一致
  4. 別人の声 → 別のキー
  5. ラベルが取り違えている → 補正
  6. 短い発話（min_sec 未満・short_floor 以上）の厳格照合
  7. 相槌（count=False）→ ラベル継続
  8. 重なり発話 → 重なりスキップ
  9. エコー窓中（enroll=False）→ 照合はするが蓄積しない
 10. AI の声紋 → AI声紋一致
 11. 音声が短すぎる（short_floor 未満）→ 照合なし
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from das.asr.live._voice_profiles import SR, VoiceProfiles

GOLDEN = Path(__file__).parent / "fixtures" / "classify_golden.json"

# 声の「向き」。同じタグなら同じ声、違えば別人（内積で分離する）。
_VOICES = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0, 0.0),
    "C": (0.0, 0.0, 1.0, 0.0),
    "AI": (0.0, 0.0, 0.0, 1.0),
    "AB": (0.72, 0.69, 0.0, 0.0),   # A寄りだが紛らわしい（2位との差が出ない）
}


def _wav(tag: str, sec: float) -> np.ndarray:
    """先頭サンプルに声の識別子を埋めた音声（埋め込みはここから作る）."""
    a = np.zeros(int(SR * sec), dtype=np.float32)
    a[:] = 0.1
    a[0] = list(_VOICES).index(tag) + 1
    return a


def _embedder(wav: np.ndarray) -> np.ndarray:
    tag = list(_VOICES)[round(float(wav[0])) - 1]
    v = np.array(_VOICES[tag], dtype=np.float64)
    return v / np.linalg.norm(v)


# (音声のタグ, 秒, STTラベル, count, overlapped, enroll, 文字数)
_SCRIPT: list[tuple] = [
    ("A", 2.0, "1", True, False, True, 40),     # 未知の声（蓄積開始）
    ("A", 2.0, "1", True, False, True, 40),     # 続けて同じ声（登録へ）
    ("A", 2.0, "1", True, False, True, 60),     # 自動登録に達する
    ("A", 2.0, "1", True, False, True, 40),     # 登録済みの声に一致
    ("B", 2.0, "2", True, False, True, 40),     # 別人（蓄積開始）
    ("B", 2.0, "2", True, False, True, 60),
    ("B", 2.0, "2", True, False, True, 60),
    ("A", 2.0, "2", True, False, True, 40),     # ラベル2なのにAの声 → 補正
    ("A", 0.6, "1", True, False, True, 10),     # 短い発話の厳格照合
    ("AB", 0.6, "1", True, False, True, 10),    # 紛らわしい短発話 → 決められない
    ("A", 2.0, "1", False, False, True, 4),     # 相槌（照合しない）
    ("B", 2.0, "1", True, True, True, 40),      # 重なり発話
    ("B", 2.0, "2", True, False, False, 40),    # エコー窓中（蓄積しない）
    ("AI", 2.0, "3", True, False, True, 40),    # AIの声
    ("C", 0.2, "4", True, False, True, 6),      # 短すぎる（照合なし）
    ("A", 2.0, "2", True, False, True, 40),     # ラベル2が2人に割れる…
    ("B", 2.0, "2", True, False, True, 40),
    ("AB", 2.0, "2", True, False, True, 40),    # …決められない → ラベル不純
]


def _run() -> list[dict]:
    vp = VoiceProfiles(path="/tmp/does-not-exist-voices.json", model="redimnet",
                       embedder=_embedder)
    vp.profiles["__AI__"] = _embedder(_wav("AI", 1.0))
    vp._active_keys.add("__AI__")
    out = []
    for tag, sec, sp, count, ov, enroll, chars in _SCRIPT:
        key = vp.classify(_wav(tag, sec), sp, overlapped=ov, count=count,
                          chars=chars, enroll=enroll)
        out.append({
            "in": [tag, sec, sp, count, ov, enroll, chars],
            "key": key,
            "last": vp.last,
            "sp_map": dict(vp.sp_map),
            "profiles": sorted(vp.profiles),
            "pool": len(vp.pool),
        })
    return out


def _canonical(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _canonical(obj.item())
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    return obj


def test_classify_output_matches_golden() -> None:
    got = _canonical(_run())
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        pytest.skip(f"期待値を作成しました: {GOLDEN}")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert got == want, (
        "声紋判定の出力が変わった。意図した変更なら "
        f"{GOLDEN.name} を作り直し、差分を必ず目で確認すること")


def test_script_covers_the_kinds_it_claims() -> None:
    """台本が主要な kind を実際に通っていること（覆っていない台本は無意味）."""
    kinds = {r["last"]["kind"] for r in _run() if r["last"]}
    for expected in ("蓄積中", "声紋一致", "補正", "ラベル継続",
                     "重なりスキップ", "AI声紋一致", "照合なし",
                     "ラベル不純"):
        assert expected in kinds, f"{expected} を通っていない: {sorted(kinds)}"
