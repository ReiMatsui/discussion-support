"""話者分離の統合・評価ロジックのテスト."""
from __future__ import annotations

import json
from typing import Any, cast

import click

from das.asr.live._constants import UNSURE_SPEAKER
from das.asr.live._diarization import (
    DiarizationEvent,
    SpeakerResolver,
    TimeSegment,
    has_overlapping_speakers,
)
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState


def test_resolver_prefers_high_confidence_voiceprint() -> None:
    resolver = SpeakerResolver()

    got = resolver.resolve(
        utterance=TimeSegment(1000, 3000),
        stt_speaker="#1",
        diarization_events=[
            DiarizationEvent(900, 3100, "SPEAKER_02", "pyannote"),
        ],
        voiceprint_speaker="田中",
        voiceprint_confidence=0.92,
    )

    assert got.speaker == "田中"
    assert got.source == "voiceprint"
    assert got.reason == "voiceprint_high_confidence"


def test_resolver_uses_diarization_when_overlap_is_large() -> None:
    resolver = SpeakerResolver()

    got = resolver.resolve(
        utterance=TimeSegment(1000, 3000),
        stt_speaker="#1",
        diarization_events=[
            DiarizationEvent(900, 2400, "SPEAKER_02", "pyannote"),
            DiarizationEvent(2400, 2800, "SPEAKER_03", "pyannote"),
        ],
    )

    assert got.speaker == "SPEAKER_02"
    assert got.source == "pyannote"
    assert got.confidence == 0.75
    assert got.reason == "diarization_overlap_0.75"


def test_resolver_falls_back_to_stt_when_all_signals_are_weak() -> None:
    resolver = SpeakerResolver()

    got = resolver.resolve(
        utterance=TimeSegment(1000, 3000),
        stt_speaker="#1",
        diarization_events=[
            DiarizationEvent(1000, 1800, "SPEAKER_02", "pyannote"),
        ],
        voiceprint_speaker="田中",
        voiceprint_confidence=0.40,
    )

    assert got.speaker == "#1"
    assert got.source == "stt"
    assert got.reason == "fallback_stt_label"


def test_resolver_accepts_short_boundary_shifted_diarization_overlap() -> None:
    resolver = SpeakerResolver()

    got = resolver.resolve(
        utterance=TimeSegment(1000, 2000),
        stt_speaker="#1",
        diarization_events=[
            DiarizationEvent(750, 1300, "SPEAKER_02", "pyannote"),
        ],
    )

    assert got.speaker == "SPEAKER_02"
    assert got.source == "pyannote"
    assert got.reason == "diarization_overlap_0.55"


def test_liveargs_and_cli_have_diarization_option() -> None:
    from das.asr.live import main
    from das.asr.live._bootstrap import LiveArgs

    assert LiveArgs().model == "stt-rt-v5"
    assert LiveArgs().diarization == "none"
    assert LiveArgs(diarization="pyannote").diarization == "pyannote"
    assert LiveArgs(
        diarization="pyannote",
        diarization_max_speakers=3,
    ).diarization_max_speakers == 3
    for param in main.params:
        if param.name == "diarization":
            choice_type = cast(click.Choice, param.type)
            assert set(choice_type.choices) == {"none", "pyannote"}
            break
    else:
        raise AssertionError("diarization option not found")


def test_recv_loop_uses_diarization_when_voiceprint_is_unavailable() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.diarization_events = [
        DiarizationEvent(900, 3100, "SPEAKER_00", "pyannote"),
    ]
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records == [{
        "ms": 1000,
        "end_ms": 3000,
        "speaker": "@diar:1",
        "text": "これはテストです",
        "diarization_raw_speaker": "SPEAKER_00",
        "speaker_source": "pyannote",
        "speaker_confidence": 1.0,
        "speaker_reason": "diarization_overlap_1.00",
    }]
    assert state.disp_name(state.records[0]["speaker"]) == "参加者A"
    assert state.names == {}


def test_recv_loop_prefers_internal_voiceprint_over_external_diarization() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    class Tracker:
        def __init__(self) -> None:
            self.last = {
                "kind": "合流",
                "label": "1",
                "name": "人物1",
                "rename": None,
            }

        def classify(self, *args: object, **kwargs: object) -> str:
            return "人物1"

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        tracker=Tracker(),  # type: ignore[arg-type]
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.asr_pcm_buf = bytearray(b"\0" * 16000 * 2 * 3)
    state.diarization_events = [
        DiarizationEvent(900, 3100, "SPEAKER_00", "pyannote"),
    ]
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records == [{
        "ms": 1000,
        "end_ms": 3000,
        "speaker": "人物1",
        "text": "これはテストです",
        "speaker_source": "voiceprint",
        "speaker_confidence": 1.0,
        "speaker_reason": "voiceprint_high_confidence",
    }]


def test_recv_loop_auto_registration_message_uses_display_label() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    class Tracker:
        def __init__(self) -> None:
            self.last = {
                "kind": "自動登録",
                "label": "1",
                "name": "人物1",
                "rename": ("#1", "人物1"),
            }

        def classify(self, *args: object, **kwargs: object) -> str:
            return "人物1"

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        tracker=Tracker(),  # type: ignore[arg-type]
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.asr_pcm_buf = bytearray(b"\0" * 16000 * 2 * 3)
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records[0]["sys"] == (
        "この声を「参加者A」として追跡開始"
        "（名前は右側の登録欄から設定できます）"
    )
    assert "人物1" not in state.records[0]["sys"]
    assert state.records[1]["speaker"] == "人物1"
    assert state.disp_name(state.records[1]["speaker"]) == "参加者A"


def test_recv_loop_normalizes_stt_label_when_diarization_is_enabled_but_unresolved() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    class Provider:
        name = "fake"

        def start(self) -> None:
            pass

        def send_audio(self, pcm16k: bytes) -> None:
            pass

        def drain_events(self) -> list[DiarizationEvent]:
            return []

        def active_events(self) -> list[DiarizationEvent]:
            return []

        def close(self) -> None:
            pass

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
        diarization_provider=Provider(),
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records == [{
        "ms": 1000,
        "end_ms": 3000,
        "speaker": "@diar:1",
        "text": "これはテストです",
        # 生のSTTラベル。以前は写像済みキー(#1)が入っており、ラベル継続中は
        # 人物N がここに入って "stt:人物N" に新規席が鋳造されていた
        # （レビュー 2026-07-30 の修正で生ラベルに戻した）。
        "stt_raw_speaker": "1",
        "speaker_source": "stt_fallback",
        "speaker_confidence": 0.0,
        "speaker_reason": "diarization_no_confident_overlap_stt_fallback",
    }]
    assert state.disp_name(state.records[0]["speaker"]) == "参加者A"


def test_recv_loop_missing_stt_speaker_becomes_unsure() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = None  # type: ignore[assignment]
    loop.cur_text = "これは誰かの発話です"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records[0]["speaker"] == UNSURE_SPEAKER
    assert "#None" not in state.anonymous_labels


def test_recv_loop_missing_stt_speaker_not_used_as_diarization_fallback() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    class Provider:
        name = "fake"

        def start(self) -> None:
            pass

        def send_audio(self, pcm16k: bytes) -> None:
            pass

        def drain_events(self) -> list[DiarizationEvent]:
            return []

        def active_events(self) -> list[DiarizationEvent]:
            return []

        def close(self) -> None:
            pass

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
        diarization_provider=Provider(),
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = None  # type: ignore[assignment]
    loop.cur_text = "これは誰かの発話です"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records == [{
        "ms": 1000,
        "end_ms": 3000,
        "speaker": UNSURE_SPEAKER,
        "text": "これは誰かの発話です",
    }]


def test_recv_loop_returns_disconnected_on_unexpected_ws_close() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    class WS:
        def recv(self) -> str:
            raise RuntimeError("closed")

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
    )
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]

    assert loop.run(WS()) == "disconnected"  # type: ignore[arg-type]


def test_recv_loop_offsets_stt_timestamps_after_reconnect() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    class WS:
        def __init__(self) -> None:
            self.messages = iter([
                json.dumps({
                    "tokens": [
                        {"text": "再接続後です", "is_final": True,
                         "speaker": "1", "start_ms": 100, "end_ms": 800},
                        {"text": "<end>"},
                    ],
                    "finished": True,
                }),
            ])

        def recv(self) -> str:
            return next(self.messages)

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.stt_time_offset_ms = 12000
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]

    assert loop.run(WS()) == "finished"  # type: ignore[arg-type]
    assert state.records[0]["ms"] == 12100
    assert state.records[0]["end_ms"] == 12800


# --- has_overlapping_speakers（pyannoteハイブリッド構成の重複発話検出）--------

def test_has_overlapping_speakers_false_for_single_speaker() -> None:
    events = [DiarizationEvent(900, 3100, "SPEAKER_00", "pyannote")]
    assert has_overlapping_speakers(events, 1000, 3000) is False


def test_has_overlapping_speakers_true_when_two_speakers_substantially_overlap() -> None:
    events = [
        DiarizationEvent(1000, 3000, "SPEAKER_00", "pyannote"),
        DiarizationEvent(1000, 2000, "SPEAKER_01", "pyannote"),
    ]
    assert has_overlapping_speakers(events, 1000, 3000) is True


def test_has_overlapping_speakers_ignores_thin_overlap_below_min_ratio() -> None:
    """相槌程度の薄い重なり(min_ratio未満)は重複発話とみなさない."""
    events = [
        DiarizationEvent(1000, 3000, "SPEAKER_00", "pyannote"),
        DiarizationEvent(2950, 3000, "SPEAKER_01", "pyannote"),  # 50ms/2000ms = 2.5%
    ]
    assert has_overlapping_speakers(events, 1000, 3000, min_ratio=0.2) is False


# --- pyannoteハイブリッド構成: クラスタ単位声紋名前付け (RecvLoop配線) ---------
# 設計: docs/design/pyannote_live1_trial_2026-07-09.md §8.4/§9

class _ClusterNamingTracker:
    """STTラベル単位のclassify()は使わせず、常にUNSURE_SPEAKERへ落として
    外部diarizationの解決結果（cluster_namer経由）だけをテストする最小スタブ."""

    def __init__(self) -> None:
        self.last = None

    def classify(self, *args: object, **kwargs: object) -> str:
        return UNSURE_SPEAKER


class _FakeClusterNamer:
    """ClusterVoiceNamer自体のロジックはtest_cluster_naming.pyで検証済みのため、
    ここではRecvLoop側の配線（sp_id決定・rekey・rec_extra）だけを見る。"""

    def __init__(self, name: str | None) -> None:
        self.name = name
        self.calls: list[tuple[str, bool]] = []

    def observe(self, raw_cluster: str, wav, *, overlapped: bool = False) -> str | None:
        self.calls.append((raw_cluster, overlapped))
        return self.name

    def rename_confirmed(self, old: str, new: str) -> None:
        return None   # rekey からの伝搬（review P2）。フェイクでは何もしない


def _make_cluster_naming_state(cluster_namer):
    import datetime

    class args_cls:                     # noqa: N801  引数クラス（値の入れ物）
        lang = "ja"
        vp_debug = False

    state = SessionState(  # type: ignore[no-untyped-call]
        args=args_cls(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        tracker=_ClusterNamingTracker(),  # type: ignore[arg-type]
        serve=False,
        cluster_namer=cluster_namer,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.asr_pcm_buf = bytearray(b"\0" * 16000 * 2 * 3)
    return state, args_cls


def test_recv_loop_cluster_naming_confirms_name_and_retroactively_renames() -> None:
    """クラスタが確定名に達したら以後その名前に帰属し、過去の@diar:Nも遡及リネームする
    （設計点4: 既存rekey機構を使った低コストな遡及リネーム）."""
    class Backend:
        def parse_message(self, raw, lang):
            return raw

    namer = _FakeClusterNamer(name="田中")
    state, args_cls = _make_cluster_naming_state(namer)
    state.diarization_events = [DiarizationEvent(900, 3100, "SPEAKER_00", "pyannote")]
    # 過去にこのクラスタへ既に@diar:1が発行済み、という状況を再現する。
    state.diarization_speaker_keys["pyannote:SPEAKER_00"] = "@diar:1"
    state.records.append({"ms": 0, "end_ms": 900, "speaker": "@diar:1", "text": "前の発話"})

    loop = RecvLoop(state, args_cls(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert namer.calls == [("pyannote:SPEAKER_00", False)]
    # 過去のレコードも遡及リネームされる。
    assert state.records[0]["speaker"] == "田中"
    new_rec = state.records[1]
    assert new_rec["speaker"] == "田中"
    assert new_rec["speaker_source"] == "cluster_voiceprint"
    assert new_rec["speaker_confidence"] == 1.0
    assert new_rec["speaker_reason"] == "pyannote_cluster_voiceprint_confirmed"


def test_recv_loop_cluster_naming_falls_back_when_unconfirmed() -> None:
    """confidence不足でクラスタが未確定の間は、従来どおりkey_for_diarization_speaker
    の匿名キー付与にフォールバックする（挙動を壊さない）."""
    class Backend:
        def parse_message(self, raw, lang):
            return raw

    namer = _FakeClusterNamer(name=None)
    state, args_cls = _make_cluster_naming_state(namer)
    state.diarization_events = [DiarizationEvent(900, 3100, "SPEAKER_00", "pyannote")]

    loop = RecvLoop(state, args_cls(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert namer.calls == [("pyannote:SPEAKER_00", False)]
    rec = state.records[0]
    assert rec["speaker"] == "@diar:1"
    assert rec["speaker_source"] == "pyannote"
    assert rec["speaker_reason"] == "diarization_overlap_1.00"


def test_recv_loop_cluster_naming_marks_overlap_region_as_unsure() -> None:
    """重複発話（複数クラスタが同時にこの区間を占める）は安全側で未確定にする
    （設計点5）。クラスタバッファも汚染しないため observe には overlapped=True で渡す."""
    class Backend:
        def parse_message(self, raw, lang):
            return raw

    namer = _FakeClusterNamer(name=None)
    state, args_cls = _make_cluster_naming_state(namer)
    state.diarization_events = [
        DiarizationEvent(1000, 3000, "SPEAKER_00", "pyannote"),
        DiarizationEvent(1000, 2000, "SPEAKER_01", "pyannote"),
    ]

    loop = RecvLoop(state, args_cls(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert namer.calls == [("pyannote:SPEAKER_00", True)]
    rec = state.records[0]
    assert rec["speaker"] == UNSURE_SPEAKER
    assert rec["speaker_source"] == "cluster_overlap"
    assert rec["speaker_confidence"] == 0.0
    assert rec["speaker_reason"] == "multiple_diarization_speakers_overlap"


def test_recv_loop_without_cluster_namer_keeps_legacy_behavior() -> None:
    """cluster_namer未設定（従来のpyannote単独モード）は挙動が変わらない."""
    class Backend:
        def parse_message(self, raw, lang):
            return raw

    state, args_cls = _make_cluster_naming_state(None)
    state.diarization_events = [DiarizationEvent(900, 3100, "SPEAKER_00", "pyannote")]

    loop = RecvLoop(state, args_cls(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    rec = state.records[0]
    assert rec["speaker"] == "@diar:1"
    assert rec["speaker_source"] == "pyannote"


def test_stt_fallback_uses_the_raw_label_not_the_mapped_key() -> None:
    """STTフォールバックのキーは生ラベルから作る（写像済みキーだと人物が分裂）.

    ラベル継続中の sp_id は 人物N（写像済みキー）でありうる。それを
    key_for_stt_fallback_speaker に渡すと台帳キー "stt:人物N" に新規 @diar:N が
    鋳造され、diarization の瞬断だけで同一人物が2席に分裂する
    （レビュー 2026-07-30）。
    """
    from das.asr.live._attribution import decide_speaker

    calls = []

    class _S:
        diarization_provider = object()

        class speaker_resolver:  # noqa: N801
            @staticmethod
            def resolve(**kw):
                class R:
                    source = "stt"
                    speaker = kw["stt_speaker"]
                    confidence = 0.0
                    reason = "fallback"
                return R()

        @staticmethod
        def diarization_window(a, b):
            return []

        @staticmethod
        def key_for_stt_fallback_speaker(speaker, duration_ms=0):
            calls.append(speaker)
            return "@diar:1"

    rec: dict = {}
    got = decide_speaker(_S(), sp_id="人物2", d={"kind": "ラベル継続"}, wav=None,
                         start_ms=0, end_ms=1000, rec_extra=rec,
                         vp_debug=False, stt_label="2")
    assert calls == ["2"], "生ラベルではなく写像済みキーで席を引いている"
    assert rec["stt_raw_speaker"] == "2"
    assert got == "@diar:1"
