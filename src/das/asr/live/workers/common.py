"""常駐処理の責務: common。会話の判定規則・時刻条件は元実装を維持する。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass







def _resilient(fn):
    """LLMワーカーを、予期しない例外1回で恒久停止させないためのラッパ.

    ワーカーのループ本体には LLM 呼び出し・state.save()（ディスクI/O）が
    あり、一過性の OSError 1回でスレッドが死ぬと、その機能は残り時間ずっと
    黙って止まる（§43 型の静かな全滅。レビュー 2026-07-30）。例外を握り
    つぶすのではなく、理由を出してから5秒後に再入する。停止要求では再入
    しない。ワーカーの進行状態（カーソル等）は state 側にあるため、再入で
    失われない。
    """
    import functools
    import traceback

    @functools.wraps(fn)
    def run(state, *args, **kwargs):
        while not state.stop.is_set():
            try:
                fn(state, *args, **kwargs)
                return
            except Exception:
                traceback.print_exc()
                print(f"# [{fn.__name__}] 予期しないエラー。5秒後に再開します",
                      flush=True)
                state.stop.wait(5.0)
    return run

