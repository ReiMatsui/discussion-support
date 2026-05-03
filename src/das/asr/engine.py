"""WhisperLiveKit の ``TranscriptionEngine`` 薄ラッパ。

モデルロードは数百MB〜1.5GB のメモリと数秒の起動時間を食うので、
プロセス内に 1 つだけ作って使い回す。

設計判断:
  - whisperlivekit / torch は重いので **モジュールトップで import しない**
    (関数内で遅延 import)。これにより、``[asr]`` extras なしで本体テストを
    回しても import エラーにならない
  - 設定の既定値は ``Settings`` から取る (大抵のユースで CLI フラグなしで動く)
  - シングルトンは ``reset_engine()`` で破棄できる (テスト・モデル切替用)
"""

from __future__ import annotations

from typing import Any

from das.logging import get_logger
from das.settings import get_settings

_log = get_logger("das.asr.engine")
_engine: Any = None


def build_engine(
    *,
    model: str | None = None,
    backend: str | None = None,
    language: str | None = None,
    diarization: bool = False,
) -> Any:
    """新しい ``TranscriptionEngine`` を作って返す (キャッシュしない)。

    引数を省略した場合は ``Settings`` の既定値 (大抵 mlx-whisper / large-v3 / ja)
    を使う。話者ダイアライゼーションは初期実装ではオフ。
    """

    # `[asr]` extras 経由で入る重い依存。遅延 import。
    from whisperlivekit import TranscriptionEngine

    settings = get_settings()
    model = model or settings.asr_model
    backend = backend or settings.asr_backend
    language = language or settings.asr_language

    _log.info(
        "asr.engine.build",
        model=model,
        backend=backend,
        language=language,
        diarization=diarization,
    )
    return TranscriptionEngine(
        model_size=model,
        backend=backend,
        lan=language,
        diarization=diarization,
    )


def get_engine(
    *,
    model: str | None = None,
    backend: str | None = None,
    language: str | None = None,
    diarization: bool = False,
) -> Any:
    """プロセス内シングルトンの ``TranscriptionEngine`` を返す。

    初回呼び出し時に与えた引数でロードされ、以後は引数を無視して同じ
    インスタンスを返す。設定を変えたいときは ``reset_engine()`` を挟む。
    """

    global _engine
    if _engine is None:
        _engine = build_engine(
            model=model,
            backend=backend,
            language=language,
            diarization=diarization,
        )
    return _engine


def reset_engine() -> None:
    """主にテスト用。次回 ``get_engine`` で再ロードされる。"""

    global _engine
    _engine = None


__all__ = ["build_engine", "get_engine", "reset_engine"]
