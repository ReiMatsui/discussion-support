"""表示ラベルの割当てが、同時に走っても重複しないこと.

`disp_name` は未割当てのキーに「参加者A」を割り当てる**副作用**を持つ。
割当ては「いま使われていない最小の文字」を選ぶので、

    used = set(self.anonymous_labels.values())   # ← ここを読んでから
    ...
    self.anonymous_labels[key] = label           # ← ここで書くまでに間がある

の間に別スレッドが同じ文字を取ると、**2つのキーが同じ「参加者A」を持つ**。
`disp_name` は受信スレッド（発話の確定）からも UI スレッド（画面描画・API）
からも呼ばれるので、この間隔は実際に開く。

重複すると表示が紛らわしいだけでは済まない。人数上限の判定
（`_known_human_slots`）は**表示ラベルの集合**で席を数えるため、2人が1席に
潰れて数えられ、上限を超えて席が空いているように見える。
"""
from __future__ import annotations

import datetime
import threading

from das.asr.live._session_state import SessionState


def _state() -> SessionState:
    return SessionState(
        args=object(), started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md", html_path="/tmp/o.html", diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns", wav_path="/tmp/o.wav", serve=False)


def test_two_threads_do_not_get_the_same_label(monkeypatch) -> None:
    s = _state()
    inside = threading.Event()
    orig = SessionState._anonymous_suffix

    def slow_suffix(index: int) -> str:
        # 片方のスレッドを「使用中の文字を調べ終えて、まだ書いていない」
        # 状態で足止めする。ロックがあれば相手はここで待たされる。
        inside.set()
        threading.Event().wait(0.25)
        return orig(index)

    def first() -> None:
        monkeypatch.setattr(SessionState, "_anonymous_suffix",
                            staticmethod(slow_suffix))
        s.disp_name("#1")
        monkeypatch.setattr(SessionState, "_anonymous_suffix",
                            staticmethod(orig))

    t = threading.Thread(target=first)
    t.start()
    assert inside.wait(3), "割当ての最中に入れなかった"
    s.disp_name("#2")          # 相手が書き込む前に割り込む
    t.join(5)

    labels = [s.anonymous_labels["#1"], s.anonymous_labels["#2"]]
    assert len(set(labels)) == 2, f"2人が同じラベルを持った: {labels}"


def test_duplicate_label_would_hide_a_seat() -> None:
    """重複が起きると席が1つ足りなく見える（上の重複を防ぐ理由）."""
    s = _state()
    s.anonymous_labels["#1"] = "参加者A"
    s.anonymous_labels["#2"] = "参加者A"     # 重複した状態を作る
    s.records.extend([{"ms": 1, "speaker": "#1"}, {"ms": 2, "speaker": "#2"}])
    assert len(s._known_human_slots()) == 1, (
        "この前提が崩れたら、重複を防ぐ理由の説明を書き直すこと")


def test_display_name_can_be_taken_while_holding_the_lock() -> None:
    """ロックを持ったまま表示名を引ける（write_md 等がそうしている）.

    素の Lock で割当てを守ると、ここで自分自身を待って止まる。RLock である
    ことの確認。
    """
    s = _state()
    done = threading.Event()

    def body() -> None:
        with s.state_lock:
            s.disp_name("#1")
        done.set()

    t = threading.Thread(target=body, daemon=True)
    t.start()
    assert done.wait(3), "ロック内から disp_name を呼んで固まった"
