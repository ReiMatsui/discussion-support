"""名乗りの検出と帰属への反映（handoff §49.11）.

会見ドメイン（§49.10）で、遠距離マイクの声紋ぎりぎり誤一致（0.43-0.50）が
記者を既存人物へ誤マージした。名乗り（「TBSの寺島です」）はテキストという
独立の証拠で、チャネル劣化の影響を受けない。

守るべき性質:

  - 検出器: 会見の名乗り5形式を拾い、雑談の「〜の◯◯です」型で誤発火しない
    （実測: 81ランで名乗り5/5検出・誤発火0）
  - 他人の紹介（さん/様/氏/君/先生+です）は名乗りではない
  - 新しい名前 → その発話の音声で名前付き登録し、その人として帰属
  - 既に同名がいる → その人として帰属（再登録しない）
  - 声紋が確定級（0.65=クラスタ確定と同じ校正線）で別人と言うなら見送る
    （司会が他人を紹介する形への防波堤）
  - 名乗り適用時は席の決め直し・門番に回さない
"""
from __future__ import annotations

import datetime

import numpy as np

from das.asr.live._nanori import detect_nanori
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState

# --- 検出器 -----------------------------------------------------------

def test_detects_kaiken_introductions():
    """会見の実発話（ラン2026-08-01_1814）5件を全て検出する."""
    assert detect_nanori(
        "え、管理者の三軒新聞の上潮と申します。冒頭、知事よりご発言よろしく"
    ) == "上潮"
    assert detect_nanori(
        "よろしくお願いいたします。TBSの寺島です。2問お伺いできれば") == "寺島"
    assert detect_nanori("日本テレビの柳原です。よろしくお願いします。") == "柳原"
    assert detect_nanori("東京新聞の奥野です。2点お願いします。") == "奥野"
    assert detect_nanori(
        "東京テレビの山田誠太郎です。よろしくお願いいたします。") == "山田誠太郎"


def test_detects_prefix_style_company_introductions():
    """前置型の会社表記（コテンラジオ #4/#6 の実発話, §49.13）.

    接尾語型（東京新聞の…）と違い「株式会社+社名+の+名前」の並び。
    法人格の前置語に限って許す（拡張後83ランで誤発火0を再測定済み）。
    """
    assert detect_nanori("はい、株式会社古典の深井龍之介です。") == "深井龍之介"
    assert detect_nanori("はい、えー、同じく株式会社古典の楊栄志です。") == "楊栄志"


def test_does_not_fire_on_casual_speech():
    """雑談・討論の「〜の◯◯です」型（素朴な正規表現で70件誤発火した実例）."""
    for tx in (
        "で、恋の話なんですけど、私の話ではなく、もちろん。",
        "トピックは腹の立つ話です",
        "あちらのお客様からです。",
        "新宿のあれですか。",
        "日本人のいいことなんですね。",
        "はい、こんばんは。延長戦です。",
        "幹事社からの質問は以上です。質問がある社は挙手の上",
        "当たり前のことですね。",
        "Um, so I think we should move on.",
    ):
        assert detect_nanori(tx) is None, tx


def test_introducing_someone_else_is_not_nanori():
    """「◯◯さんです」等は他人の紹介なので拾わない."""
    assert detect_nanori("続いては、東京大学の田中さんです。") is None
    assert detect_nanori("日本テレビの柳原様です。") is None


# --- 帰属への反映 ------------------------------------------------------

class _Args:
    lang = "ja"
    vp_debug = False


class _Backend:
    def parse_message(self, raw, lang):
        return raw


class _NanoriTracker:
    """名乗りテスト用フェイク: 声紋はぎりぎりの誤一致（0.45）を返す."""

    def __init__(self, *, sim=0.45, kind="声紋一致", profiles=()) -> None:
        self.last = {"kind": kind, "label": "2", "name": "人物2", "sim": sim}
        self.enrolled: list[str] = []
        self._profiles = list(profiles)

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        return "人物2"

    def active_profile_names(self):
        return list(self._profiles) + self.enrolled

    def enroll_from_audio(self, name, wav):
        self.enrolled.append(name)
        return True


def _make_state(tmp_path, tracker):
    state = SessionState(  # type: ignore[no-untyped-call]
        args=_Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp_path / "o.md"),
        html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"),
        turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"),
        tracker=tracker,
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.asr_pcm_buf = bytearray(np.full(16000 * 10, 12000, dtype="<i2").tobytes())
    state.cluster_namer = object()   # ハイブリッド構成の印
    return state


def _flush(tmp_path, tracker, text):
    state = _make_state(tmp_path, tracker)
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "2"
    loop.cur_text = text
    loop.cur_ms, loop.cur_end = 1000, 5000
    loop.flush()  # type: ignore[no-untyped-call]
    return state


def test_new_name_is_enrolled_and_attributed(tmp_path):
    """新しい名前の名乗り: 声紋のぎりぎり誤一致（人物2, 0.45）を上書きして
    名乗った本人として登録・帰属する（§49.10 の誤マージの直撃対策）."""
    tracker = _NanoriTracker()
    state = _flush(tmp_path, tracker,
                   "東京新聞の奥野です。2点お願いします。1点目、副知事の交代について")
    assert tracker.enrolled == ["奥野"]
    assert state.records[-1]["speaker"] == "奥野"
    assert state.records[-1]["speaker_source"] == "nanori"
    sys_texts = [str(r.get("sys")) for r in state.records if "sys" in r]
    assert any("奥野" in t for t in sys_texts)   # 追跡開始の告知


def test_reintroduction_reuses_existing_profile(tmp_path):
    """既に同名がいる場合は再登録せず、その人として帰属する."""
    tracker = _NanoriTracker(profiles=["奥野"])
    state = _flush(tmp_path, tracker, "東京新聞の奥野です。改めてもう1点伺います。")
    assert tracker.enrolled == []
    assert state.records[-1]["speaker"] == "奥野"


def test_strong_voiceprint_match_vetoes_nanori(tmp_path):
    """声紋が確定級（>=0.65）で既存の別人と言うなら名乗りを見送る
    （司会が他人を「◯◯です」と紹介する形への防波堤）."""
    tracker = _NanoriTracker(sim=0.80)
    state = _flush(tmp_path, tracker, "日本テレビの柳原です。よろしくお願いします。")
    assert tracker.enrolled == []
    assert state.records[-1]["speaker"] != "柳原"


def test_nanori_bypasses_impure_guard(tmp_path):
    """名乗り適用時は門番（§47）に回さない——長い発話×低simでも本人扱い."""
    tracker = _NanoriTracker(sim=0.30, kind="ラベル不純")
    state = _flush(tmp_path, tracker,
                   "東京テレビの山田誠太郎です。よろしくお願いいたします。1点伺います。"
                   "今週、都議会が閉会しまして補正予算が成立しました")
    assert state.records[-1]["speaker"] == "山田誠太郎"
    assert state.records[-1]["speaker_source"] == "nanori"
