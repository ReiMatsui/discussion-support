"""write_html の出力を丸ごと固定する（分割リファクタの安全網）.

サイドバー各パネル（話者・プロファイル・発言量・論点・エージェント）と
本文（sys行・補正バッジ含む）を全部含む状態を1つ作り、生成HTMLの一致を
フィクスチャで固定する。分割で1文字でも変われば落ちる。
"""
from __future__ import annotations

import datetime
import pathlib
import threading
from typing import ClassVar

from das.asr.live._session_state import SessionState

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "write_html_golden.html"


class _Args:
    lang = "ja"
    vp_debug = False


class _Tracker:
    sp_map: ClassVar[dict] = {}

    def all_profile_names(self):
        return ["田中", "佐藤"]

    def active_profile_names(self):
        return ["田中"]


class _Agent:
    mode = "facilitator"
    _connected = True
    _conn_error = None
    voice = "shimmer"
    trigger_n = 10


def _state(tmp_path) -> SessionState:
    s = SessionState(  # type: ignore[no-untyped-call]
        args=_Args(),
        started=datetime.datetime(2026, 1, 2, 3, 4),
        out_path=str(tmp_path / "o.md"),
        html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"),
        turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"),
        tracker=_Tracker(),  # type: ignore[arg-type]
        serve=True,
    )
    s.agent = _Agent()
    s.records.extend([
        {"ms": 1000, "end_ms": 3000, "speaker": "人物1", "text": "こんにちは"},
        {"sys": "この声を「参加者A」として追跡します"},
        {"ms": 4000, "end_ms": 9000, "speaker": "人物2", "text": "よろしく",
         "vp": "補正", "note": "声紋補正の理由"},
        {"ms": 9500, "end_ms": 9900, "speaker": "?", "text": "ほい"},
    ])
    with s.topics_lock:
        s.topics.append({"topic": "旅行の計画", "speaker": "人物1"})
    return s


def test_write_html_output_is_stable(tmp_path):
    s = _state(tmp_path)
    s.write_html(live=False)
    got = pathlib.Path(s.html_path).read_text(encoding="utf-8")
    if not FIXTURE.exists():          # 初回だけ焼き付け
        FIXTURE.write_text(got, encoding="utf-8")
    assert got == FIXTURE.read_text(encoding="utf-8"), \
        "write_html の出力が変わった（意図した変更なら fixture を消して再生成）"


def test_write_html_is_thread_safe_entry(tmp_path):
    """live=True でも書けること（別スレッドからの呼び出し想定の入口確認）."""
    s = _state(tmp_path)
    t = threading.Thread(target=s.write_html)
    t.start()
    t.join(timeout=5)
    assert pathlib.Path(s.html_path).exists()
