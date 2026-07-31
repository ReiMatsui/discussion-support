"""flush の出力を丸ごと固定する検証台（分割・整理の安全網）.

**なぜ要るのか**: 帰属の数字（9本平均 89.9%）は diag を読み直して採点して
いる。つまり `flush` を壊しても数字は動かず、気づけない。ここでは台本を
与えて `flush` を実際に通し、`records` と diag 行の**全内容**を固定する。
リファクタで1バイトでも変われば落ちる。

台本は帰属の主要な経路を一通り通す:

  1. 通常発話（声紋一致）
  2. 相槌 → 未確定に落ち、bc 印が付く
  3. 別人の発話（声紋一致）
  4. 補正（声紋がSTTラベルの取り違えを直す）
  5. 自動登録（人物を鋳造。席が空いていれば追跡開始を告げる）
  6. ラベル不純 → 席の実音声で決め直す
  7. 蓄積中で裏付けなし → 未確定
  8. 席上限で落ちる発話 → 席の音声で寄せ直す
  9. テキスト安全網によるエコー破棄
 10. 声紋によるエコー破棄
 11. STT が話者を返さない発話
 12. 空テキスト（何も起きない）

期待値は `fixtures/flush_golden.json`。**中身の良し悪しは問わない**
（それは各機能のテストの仕事）。ここが守るのは「変えていないつもりの変更で
出力が変わらないこと」だけ。
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from das.asr.live._cluster_naming import ClusterVoiceNamer
from das.asr.live._diarization import DiarizationEvent, SpeakerResolver
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._seat_audio import SeatAudio
from das.asr.live._session_state import SessionState

GOLDEN = Path(__file__).parent / "fixtures" / "flush_golden.json"
SR = 16000


class _Args:
    lang = "ja"
    vp_debug = False
    diarization = "pyannote"
    vp_cluster_naming = True
    diarization_max_speakers = 3


class _Backend:
    def parse_message(self, raw, lang):
        return raw


class _ScriptedTracker:
    """台本どおりに判定を返す声紋トラッカー.

    `embed_audio` は音声の先頭サンプルの値で人物を決める（値が同じなら同じ
    埋め込み）。席の参照・クラスタの蓄積が意味を持つようにするため。
    """

    def __init__(self, script: list[dict]) -> None:
        self._script = script
        self._i = 0
        self.last: dict | None = None
        self.profiles: dict = {}
        self.hybrid = False

    def set_hybrid(self, on: bool) -> None:
        self.hybrid = on

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        step = self._script[self._i]
        self._i += 1
        self.last = dict(step["last"]) if step.get("last") is not None else None
        return step["key"]

    def embed_audio(self, wav):
        if wav is None or len(wav) == 0:
            return None
        tag = round(float(wav[0]) * 100)
        v = np.zeros(4, dtype=np.float64)
        v[abs(tag) % 4] = 1.0
        return v


def _state(tmp_path: Path, tracker) -> SessionState:
    seat = SeatAudio(tracker, ref_sec=30.0, min_ref_sec=1.0)
    s = SessionState(
        args=_Args(), started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp_path / "o.md"), html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"), turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"), tracker=tracker, serve=False,
        speaker_resolver=SpeakerResolver(),
        cluster_namer=ClusterVoiceNamer(tracker),
        seat_audio=seat)
    s.save = lambda *a, **k: None            # type: ignore[method-assign]
    # diarization は供給済みの区間を直接積む（provider は空を返すだけ）。
    s.diarization_provider = _EmptyProvider()  # type: ignore[assignment]
    return s


class _EmptyProvider:
    """区間はテスト側が積むので、provider は何も返さない."""

    def drain_events(self) -> list:
        return []


# 台本。tag は音声の中身（＝誰の声か）。cls は classify の返り値と診断。
_SCRIPT: list[dict] = [
    {"label": "1", "text": "今日の目的を確認します", "ms": 1000, "end": 5000,
     "tag": 0.1, "diar": ("SPEAKER_00", 1000, 5000),
     "cls": {"key": "人物1", "last": {"kind": "声紋一致", "label": "1",
                                     "name": "人物1", "sim": 0.82}}},
    {"label": "1", "text": "はい", "ms": 5200, "end": 5600,
     "tag": 0.1, "diar": ("SPEAKER_00", 5200, 5600),
     "cls": {"key": "人物1", "last": {"kind": "声紋一致", "label": "1",
                                     "name": "人物1", "sim": 0.80}}},
    {"label": "2", "text": "では私から報告します", "ms": 6000, "end": 11000,
     "tag": 0.2, "diar": ("SPEAKER_01", 6000, 11000),
     "cls": {"key": "人物2", "last": {"kind": "声紋一致", "label": "2",
                                     "name": "人物2", "sim": 0.79}}},
    {"label": "2", "text": "先週の数字はこうでした", "ms": 11500, "end": 15500,
     "tag": 0.1, "diar": ("SPEAKER_00", 11500, 15500),
     "cls": {"key": "人物1", "last": {"kind": "補正", "label": "2", "prev": "人物2",
                                     "name": "人物1", "sim": 0.77}}},
    {"label": "3", "text": "こちらでも確認しています", "ms": 16000, "end": 21000,
     "tag": 0.3, "diar": ("SPEAKER_02", 16000, 21000),
     "cls": {"key": "人物3", "last": {"kind": "自動登録", "label": "3",
                                     "name": "人物3", "rename": None,
                                     "sim": 0.74}}},
    {"label": "3", "text": "補足すると条件が違います", "ms": 21500, "end": 24500,
     "tag": 0.2, "diar": ("SPEAKER_01", 21500, 24500),
     "cls": {"key": "?", "last": {"kind": "ラベル不純", "label": "3",
                                  "name": "人物2", "sim": 0.31}}},
    {"label": "4", "text": "私はまだ話していません", "ms": 25000, "end": 28000,
     "tag": 0.3, "diar": ("SPEAKER_03", 25000, 28000),
     "cls": {"key": "人物4", "last": {"kind": "蓄積中", "label": "4",
                                     "name": "人物3", "sim": 0.20}}},
    {"label": "5", "text": "四人目として発言します", "ms": 28500, "end": 32500,
     "tag": 0.1, "diar": ("SPEAKER_04", 28500, 32500),
     "cls": {"key": "人物5", "last": {"kind": "ラベル継続", "label": "5",
                                     "name": "人物5", "sim": 0.40}}},
    {"label": "1", "text": "まとめると三点あります", "ms": 33000, "end": 36000,
     "tag": 0.1, "diar": ("SPEAKER_00", 33000, 36000),
     "cls": {"key": "人物1", "last": {"kind": "声紋一致", "label": "1",
                                     "name": "人物1", "sim": 0.85}}},
    {"label": "2", "text": "エコーとして捨てられる文です", "ms": 37000, "end": 40000,
     "tag": 0.2, "diar": None, "echo_text": True, "cls": None},
    {"label": "2", "text": "AIの声が漏れ込んだ文です", "ms": 41000, "end": 44000,
     "tag": 0.2, "diar": None,
     "cls": {"key": "__AI__", "last": {"kind": "声紋一致", "label": "2",
                                       "name": "__AI__", "sim": 0.91}}},
    {"label": "", "text": "話者が付かなかった発話です", "ms": 45000, "end": 48000,
     "tag": 0.1, "diar": ("SPEAKER_00", 45000, 48000), "cls": None},
    {"label": "1", "text": "   ", "ms": 49000, "end": 49500,
     "tag": 0.1, "diar": None, "cls": None},
]


class _EchoAgent:
    """指定の1文だけをエコー扱いにするフェイク（テキスト安全網の起動用）."""

    def __init__(self, text: str) -> None:
        self.in_echo_window = True
        self.ai_speaking = False
        self._text = text

    def _best_similarity(self, text: str) -> float:
        return 0.9 if text.strip() == self._text else 0.0


def _run_script(tmp_path: Path) -> dict:
    echo_text = next(x["text"] for x in _SCRIPT if x.get("echo_text"))
    tracker = _ScriptedTracker([x["cls"] for x in _SCRIPT
                                if x["cls"] is not None])
    s = _state(tmp_path, tracker)
    s.agent = _EchoAgent(echo_text)          # type: ignore[assignment]
    # 台本の各発話区間に「誰の声か」を書いた音声を敷く。
    total = max(x["end"] for x in _SCRIPT) + 1000
    buf = np.zeros(int(total / 1000 * SR), dtype=np.float32)
    for x in _SCRIPT:
        a, b = int(x["ms"] / 1000 * SR), int(x["end"] / 1000 * SR)
        buf[a:b] = x["tag"]
    s.asr_pcm_buf = bytearray((buf * 32767).astype("<i2").tobytes())

    loop = RecvLoop(s, _Args(), _Backend())  # type: ignore[arg-type]
    for x in _SCRIPT:
        if x["diar"] is not None:
            spk, a, b = x["diar"]
            s.diarization_events.append(
                DiarizationEvent(a, b, spk, "pyannote"))
        loop.cur_speaker = x["label"]        # type: ignore[assignment]
        loop.cur_text = x["text"]
        loop.cur_ms, loop.cur_end = x["ms"], x["end"]
        loop.flush()                         # type: ignore[no-untyped-call]

    diag = [json.loads(x) for x in
            Path(s.diag_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"records": s.records, "diag": diag,
            "labels": dict(s.anonymous_labels)}


def _canonical(obj):
    """浮動小数の桁ゆらぎを潰して JSON に落とす."""
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _canonical(obj.item())
    return obj


def test_flush_output_matches_golden(tmp_path: Path) -> None:
    got = _canonical(_run_script(tmp_path))
    if not GOLDEN.exists():           # 初回だけ作る（差分はレビューで見る）
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        pytest.skip(f"期待値を作成しました: {GOLDEN}")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert got == want, (
        "flush の出力が変わった。意図した変更なら "
        f"{GOLDEN.name} を作り直し、差分を必ず目で確認すること")


def test_script_covers_the_paths_it_claims(tmp_path: Path) -> None:
    """台本が主要な経路を実際に通っていること（覆っていない台本は無意味）."""
    out = _run_script(tmp_path)
    kinds = {r.get("vp") for r in out["records"]}
    sources = {r.get("speaker_source") for r in out["records"]}
    diag_types = {d.get("type") for d in out["diag"]}
    assert "補正" in kinds
    assert "seat_assign" in sources, "席の割当てを通っていない"
    assert any(r.get("bc") for r in out["records"]), "相槌を通っていない"
    assert any(r["speaker"] == "?" for r in out["records"]), "未確定が出ていない"
    assert "echo_drop" in diag_types, "エコー破棄を通っていない"
    assert len({d.get("src") for d in out["diag"] if d.get("type") == "echo_drop"}
               & {"agent", "voiceprint"}) == 2, "エコー破棄の2経路を通っていない"
