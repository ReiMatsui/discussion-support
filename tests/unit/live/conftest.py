"""live配下の介入・割り込みロジック用テスト基盤.

実WebSocket / sounddevice を使わずに RealtimeAgent / ConversationPartner の
イベント処理・トリガー・割り込みロジックを検証するためのスタブ群。
"""
from __future__ import annotations

import base64
import json

import pytest

from das.asr.live.agents._partner import ConversationPartner
from das.asr.live.agents._realtime import RealtimeAgent


class FakeWS:
    """RealtimeAgent/Partner が使う最小限の WebSocket スタブ.

    send() で送られた JSON を解析して self.sent に記録する。
    recv() はテストでは使わない（_recv_loop スレッドは起動しないため）。
    """

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def recv(self) -> str:  # pragma: no cover - テストでは未使用
        raise RuntimeError("FakeWS.recv は使用しない想定")

    def close(self) -> None:
        self.closed = True

    # --- 検証ヘルパー ---
    def types(self) -> list[str]:
        return [m.get("type") for m in self.sent]

    def last_create_text(self) -> str | None:
        """直近の conversation.item.create の input_text を返す."""
        for m in reversed(self.sent):
            if m.get("type") == "conversation.item.create":
                content = m["item"]["content"]
                return content[0]["text"]
        return None


def make_chunk(n_samples: int = 1200) -> str:
    """ダミーPCM音声のbase64文字列（24kHz 16bit 相当）."""
    return base64.b64encode(b"\x01\x02" * n_samples).decode()


def queue_real_chunks(agent: RealtimeAgent | ConversationPartner) -> int:
    """再生キューに積まれた実音声チャンク数（None終端を除く）.

    キュー要素は (epoch, payload) のタプル。payload が None でないものを数える。
    """
    return sum(1 for (_epoch, payload) in list(agent._audio_q.queue)
               if payload is not None)


@pytest.fixture
def agent() -> RealtimeAgent:
    """FakeWS を接続済みにした facilitator エージェント（スレッド未起動）."""
    a = RealtimeAgent(api_key="test-key", mode="facilitator")
    a.ws = FakeWS()  # type: ignore[assignment]
    a._connected = True
    return a


@pytest.fixture
def partner() -> ConversationPartner:
    """FakeWS を接続済みにした対話パートナー（スレッド未起動）."""
    p = ConversationPartner(api_key="test-key", topic="テスト議題")
    p.ws = FakeWS()  # type: ignore[assignment]
    p._connected = True
    return p


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
