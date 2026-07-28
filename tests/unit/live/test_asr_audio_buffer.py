"""STT送信済みPCMバッファのテスト."""
from __future__ import annotations

import datetime
from typing import Any

from das.asr.live._audio_io import _run_sender
from das.asr.live._session_state import SessionState


class _Backend:
    def make_end_message(self, seq: int) -> str:
        return f"END:{seq}"


class _WS:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.sent: list[bytes | str] = []

    def send(self, payload: bytes | str) -> None:
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("send failed")
        self.sent.append(payload)


class _DiarizationProvider:
    name = "fake"

    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.drains = 0

    def start(self) -> None:
        pass

    def send_audio(self, pcm16k: bytes) -> None:
        self.audio.append(pcm16k)

    def drain_events(self) -> list[Any]:
        self.drains += 1
        return []

    def close(self) -> None:
        pass


def _make_state() -> SessionState:
    return SessionState(
        args=object(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
    )


def test_sender_keeps_only_successfully_sent_audio_for_asr_buffer() -> None:
    state = _make_state()
    provider = _DiarizationProvider()
    state.diarization_provider = provider
    ws = _WS(fail_first=True)
    state.stt_ws = ws  # type: ignore[assignment]
    first = b"\x01\x00" * 1600
    second = b"\x02\x00" * 1600

    state.audio_q.put(first)
    state.audio_q.put(second)
    state.audio_q.put(None)
    _run_sender(state, _Backend())  # type: ignore[arg-type]

    assert state.pcm_buf == first + second
    assert state.pcm_total_bytes == len(first) + len(second)
    assert state.asr_pcm_buf == second
    assert state.asr_pcm_total_bytes == len(second)
    assert ws.sent == [second, "END:1"]
    assert provider.audio == [second]
    assert provider.drains == 1


def test_sender_buffers_setup_audio_without_recording_or_stt_send() -> None:
    state = _make_state()
    state.waiting_to_start = True
    state.stt_ws = None
    pcm = b"\x03\x00" * 1600

    state.audio_q.put(pcm)
    state.audio_q.put(None)
    _run_sender(state, _Backend())  # type: ignore[arg-type]

    assert state.pcm_buf == pcm
    assert state.pcm_total_bytes == 0
    assert state.asr_pcm_buf == bytearray()
    assert state.asr_pcm_total_bytes == 0


def test_open_wav_resets_asr_buffer_offsets() -> None:
    state = _make_state()
    state.asr_pcm_buf.extend(b"abc")
    state.asr_pcm_buf_offset = 10
    state.asr_pcm_total_bytes = 3
    state.stt_time_offset_ms = 999
    state._stt_connection_audio_base_bytes = 123

    state.open_wav()  # type: ignore[no-untyped-call]

    assert state.asr_pcm_buf == bytearray()
    assert state.asr_pcm_buf_offset == 0
    assert state.asr_pcm_total_bytes == 0
    assert state.stt_time_offset_ms == 0
    assert state._stt_connection_audio_base_bytes == 0


def test_recording_holds_only_sent_audio_so_wav_matches_utterance_ms(
        tmp_path) -> None:
    """録音wavは「STTへ送れた音声」だけを持ち、長さが ms の原点と一致する.

    発話の ms は送信済みバイト数そのもの。送れなかったチャンクまで録音に
    混ぜると wav だけが先へずれ、後から wav を ms で切ったときに隣の話者の
    声を掴む（実測で短い発話のオラクル精度が偶然以下まで落ちた）。
    """
    import wave

    state = _make_state()
    state.wav_path = str(tmp_path / "rec.wav")
    state.open_wav()  # type: ignore[no-untyped-call]
    ws = _WS(fail_first=True)
    state.stt_ws = ws  # type: ignore[assignment]
    dropped = b"\x01\x00" * 16000        # 送信に失敗する1秒
    kept = b"\x02\x00" * (16000 * 12)    # 送れる12秒（短い録音は破棄されるため）

    state.audio_q.put(dropped)
    state.audio_q.put(kept)
    state.audio_q.put(None)
    _run_sender(state, _Backend())  # type: ignore[arg-type]
    state.finalize_wav()

    with wave.open(state.wav_path) as w:
        frames = w.readframes(w.getnframes())
    assert frames == kept, "送れなかった音声が録音に混ざっている"
    # wav の末尾位置(ms) と、次の発話に振られる ms の原点が一致すること
    assert len(frames) // 32 == state.current_asr_ms()


def test_stt_connection_start_sets_timestamp_offset() -> None:
    state = _make_state()
    state.asr_pcm_total_bytes = 16000 * 2 * 12

    state.mark_stt_connection_started()

    assert state.stt_time_offset_ms == 12000
    assert state.stt_abs_ms(345) == 12345
    assert state.stt_abs_ms(None) is None
