"""リアルタイム音声認識モジュール。

2系統のASRパイプラインが共存する:

1. WhisperLiveKit系 (engine.py / mic.py / session.py)
   - ``das listen`` コマンド (cli.py) から使用
   - Orchestrator/Bus経由でAF構築と連携
   - ``[asr]`` extras が必要

2. Soniox/Speechmatics系 (soniox_live.py + _*.py)
   - ``das listen-soniox`` / ``python -m das.asr.soniox_live`` から使用
   - リアルタイム文字起こし + AIファシリテーション + 議論モード
   - STTバックエンドはProtocolで差し替え可能 (_stt_backend.py)

本 __init__.py はWhisperLiveKit系のみをexportする。
Soniox系は ``das.asr.soniox_live`` を直接importして使用する。
"""

from __future__ import annotations

from das.asr.engine import build_engine, get_engine, reset_engine
from das.asr.session import LiveAsrSession

__all__ = [
    "LiveAsrSession",
    "build_engine",
    "get_engine",
    "reset_engine",
]
