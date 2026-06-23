"""リアルタイム音声認識モジュール。

2系統のASRパイプラインが共存する:

1. WhisperLiveKit系 (engine.py / mic.py / session.py)
   - ``das listen`` コマンド (cli.py) から使用
   - Orchestrator/Bus経由でAF構築と連携
   - ``[asr]`` extras が必要

2. リアルタイム議事録系 (live/ パッケージ)
   - ``das listen-soniox`` / ``python -m das.asr.live`` から使用
   - リアルタイム文字起こし + AIファシリテーション + 議論モード
   - STTバックエンドはProtocolで差し替え可能 (live/stt/)
   - AIエージェント群: live/agents/

本 __init__.py はWhisperLiveKit系のみをexportする。
リアルタイム議事録系は ``das.asr.live`` を直接importして使用する。
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
