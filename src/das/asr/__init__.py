"""リアルタイム音声認識 (WhisperLiveKit ラッパ)。

``[asr]`` extras を入れたときだけ実体が動く。本体テストパスから import
されないように、サブモジュールでさらに遅延 import している。
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
