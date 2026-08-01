"""--no-llm / --no-intervention（§49.9）: LLM補助の一括停止.

検証の再生ラン（YouTube素材等）でトークンを消費しないためのスイッチ。
話者帰属には一切関わらないので、成績の比較可能性は保たれる。

守るべき性質:

  - no_llm=True なら、OPENAI_API_KEY があっても LLM 常駐ワーカーを1つも
    起動しない
  - no_llm=False（既定）は従来どおり（論点抽出が動く）
  - listen-soniox --no-intervention は下層へ --no-agent と --no-llm を渡す
"""
from __future__ import annotations

import das.asr.live._bootstrap as bootstrap
from das.asr.live._bootstrap import LiveArgs, _start_llm_workers
from das.cli._listen import _NO_INTERVENTION_FLAGS


class _ThreadSpy:
    """threading.Thread の代役: 起動要求を記録するだけで実際は走らせない."""

    def __init__(self) -> None:
        self.started: list[object] = []

    def __call__(self, *a, target=None, **k):
        spy = self

        class _T:
            def start(self) -> None:
                spy.started.append(target)

        return _T()


class _State:
    agent = None


def test_no_llm_starts_no_workers_even_with_key(monkeypatch, capsys):
    spy = _ThreadSpy()
    monkeypatch.setattr(bootstrap.threading, "Thread", spy)
    _start_llm_workers(_State(), LiveArgs(no_llm=True),
                       oai_key="sk-test", oai_model="gpt-test",
                       out_path="/tmp/x.md", explicit_agenda=True)
    assert spy.started == []
    assert "--no-llm" in capsys.readouterr().out


def test_default_still_starts_topic_worker(monkeypatch):
    spy = _ThreadSpy()
    monkeypatch.setattr(bootstrap.threading, "Thread", spy)
    _start_llm_workers(_State(), LiveArgs(),
                       oai_key="sk-test", oai_model="gpt-test",
                       out_path="/tmp/x.md", explicit_agenda=True)
    assert len(spy.started) == 1   # 論点抽出（agent 無しなら他は起動しない）


def test_live_args_default_is_llm_enabled():
    assert LiveArgs().no_llm is False


def test_no_intervention_forwards_both_flags():
    assert "--no-agent" in _NO_INTERVENTION_FLAGS
    assert "--no-llm" in _NO_INTERVENTION_FLAGS
