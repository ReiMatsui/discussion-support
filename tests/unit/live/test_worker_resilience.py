"""LLMワーカーの復元ラッパのテスト."""
import threading

from das.asr.live import _workers


class _S:
    def __init__(self):
        self.stop = threading.Event()


def test_resilient_reenters_after_an_unexpected_exception(capsys):
    """予期しない例外1回で恒久停止せず、理由を出して再入する（§43型の全滅防止）."""
    s = _S()
    calls = []

    @_workers._resilient
    def worker(state):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("一過性のディスクエラー")
        state.stop.set()

    # stop.wait(5.0) を待たない
    s.stop.wait = lambda t=None: None
    worker(s)
    assert len(calls) == 2, "例外の後に再入していない"
    assert "再開します" in capsys.readouterr().out


def test_resilient_does_not_reenter_after_stop():
    s = _S()
    calls = []

    @_workers._resilient
    def worker(state):
        calls.append(1)
        state.stop.set()
        raise OSError("停止中のエラー")

    s.stop.wait = lambda t=None: None
    worker(s)
    assert len(calls) == 1
