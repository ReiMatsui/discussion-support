"""live配下の介入・割り込みロジック用テスト基盤.

実WebSocket / sounddevice を使わずにエージェントのイベント処理・トリガーの
ロジックを検証するためのスタブ群。
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fast_worker_tick(monkeypatch):
    """常駐ワーカーの周期を詰めて、テストの実時間待ちを減らす.

    ワーカーは 0.25 秒ごとに状態を見に行く。テストの多くは「数tick回れば
    判断が出る」のを待っているだけなのに、その待ちが実時間で積み上がって
    スイートの大半を占めていた（2026-07-28 の計測で上位15件＝47秒）。

    間隔を詰めても判断は変わらない——クールダウンや pause は壁時計の差で
    見ているので、覗く回数が増えても早く発火はしない。**本番の値は変えない**。
    """
    from das.asr.live import _workers
    monkeypatch.setattr(_workers, "WORKER_TICK_SEC", 0.01)
