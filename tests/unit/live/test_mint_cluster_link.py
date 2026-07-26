"""鋳造時クラスタリンク（二重帳簿の根治, opt-in）のテスト.

設計・校正: docs/design/handoff_2026-07-25_dual_ledger_rootcure.md 案B。
同じ人間がクラスタ帳簿(@diar:N)と声紋帳簿(人物N)に二重に載って席を食い潰し、
実在者を締め出す問題への対処。声紋側が新しい戸籍を鋳造した瞬間だけ、席を持つ
クラスタの蓄積声紋と**対称比較**して同一人物なら統合する。

ここで固定する性質:
  - 統合の条件（下限類似度・2位とのmargin・席の有無・蓄積量・確定済み除外）
  - 既定は無効＝従来挙動不変（opt-in フラグと cluster_namer 両方が要る）
  - 統合は SessionState.rekey を通り、台帳（diarization_speaker_keys /
    ClusterVoiceNamer._confirmed）まで揃って書き変わる
"""
from __future__ import annotations

import datetime
import json
from types import SimpleNamespace

import numpy as np

from das.asr.live._cluster_naming import ClusterVoiceNamer
from das.asr.live._constants import SR
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState

# 決め打ちの正規化ベクトル。_A と _A2 は類似 0.9（同一人物相当）、
# _B は _A と直交（別人相当）、_MID は _A と 0.6（境界確認用）。
_A = np.array([1.0, 0.0, 0.0])
_A2 = np.array([0.9, np.sqrt(1 - 0.81), 0.0])
_B = np.array([0.0, 1.0, 0.0])
_MID = np.array([0.6, 0.8, 0.0])


def _wav(seconds: float, fill: float) -> np.ndarray:
    return np.full(int(seconds * SR), fill, dtype=np.float32)


class _EmbedTracker:
    """embed_audio だけを持つ最小フェイク（波形の先頭値でベクトルを引く）."""

    def __init__(self, embed_map, profiles=None):
        self._embed_map = dict(embed_map)
        self.profiles = dict(profiles or {})

    def embed_audio(self, wav):
        if wav.size == 0:
            return None
        return self._embed_map.get(round(float(wav[0]), 3))

    def match_profile(self, wav):   # observe 経路は使わないので常に不成立
        return None


def _namer_with(buffers, embed_map, *, profiles=None, min_sec=5.0):
    """指定のクラスタ蓄積を持つ ClusterVoiceNamer を作る."""
    tracker = _EmbedTracker(embed_map, profiles)
    namer = ClusterVoiceNamer(tracker, min_sec=min_sec)
    for raw, wav in buffers.items():
        namer._buffers[raw] = [wav]
    return namer


# ---------------------------------------------------------------------------
# link_minted_profile: 統合先の選び方
# ---------------------------------------------------------------------------

def test_links_to_the_matching_seat_cluster():
    """同一人物の席持ちクラスタが下限以上なら (生ID, 席キー, 類似度) を返す."""
    namer = _namer_with({"pyannote:SPEAKER_00": _wav(6.0, 0.1)},
                        {0.1: _A2})

    hit = namer.link_minted_profile(_A, lambda raw: "@diar:1")

    assert hit is not None
    raw, key, sim = hit
    assert (raw, key) == ("pyannote:SPEAKER_00", "@diar:1")
    assert sim == np.dot(_A, _A2)


def test_no_link_when_similarity_below_floor():
    """別人（下限未満）は統合しない＝従来どおり新しい戸籍のまま."""
    namer = _namer_with({"pyannote:SPEAKER_00": _wav(6.0, 0.1)}, {0.1: _B})

    assert namer.link_minted_profile(_A, lambda raw: "@diar:1") is None


def test_no_link_when_second_best_is_close():
    """2位が僅差なら統合しない（1つのクラスタが複数人に中程度似る誤リンク対策）.

    実測でも誤リンクは「短い断片クラスタが3人全員に0.50-0.69で似る」形で
    起きており、1位が2位を明確に上回ることを要求する。
    """
    namer = _namer_with(
        {"pyannote:SPEAKER_00": _wav(6.0, 0.1),
         "pyannote:SPEAKER_01": _wav(6.0, 0.2)},
        {0.1: _A2, 0.2: _A2},   # 2クラスタが同じ類似 → margin 不足
    )

    assert namer.link_minted_profile(_A, lambda raw: "@diar:1") is None


def test_skips_cluster_without_seat():
    """席を持たないクラスタは統合先にできない（席の二重取りがまだ無い＝実害なし）."""
    namer = _namer_with({"pyannote:SPEAKER_00": _wav(6.0, 0.1)}, {0.1: _A2})

    assert namer.link_minted_profile(_A, lambda raw: None) is None


def test_skips_cluster_with_insufficient_audio():
    """蓄積が min_sec 未満のクラスタは声紋が当てにならないので候補にしない."""
    namer = _namer_with({"pyannote:SPEAKER_00": _wav(2.0, 0.1)}, {0.1: _A2},
                        min_sec=5.0)

    assert namer.link_minted_profile(_A, lambda raw: "@diar:1") is None


def test_skips_already_confirmed_cluster():
    """確定済みクラスタは既に戸籍を持つため統合の対象外."""
    namer = _namer_with({"pyannote:SPEAKER_00": _wav(6.0, 0.1)}, {0.1: _A2})
    namer._confirmed["pyannote:SPEAKER_00"] = "田中"

    assert namer.link_minted_profile(_A, lambda raw: "@diar:1") is None


def test_adopt_confirmed_short_circuits_future_observes():
    """統合を書き込むと以後の observe が統合先を短絡で返し、蓄積は解放される."""
    namer = _namer_with({"pyannote:SPEAKER_00": _wav(6.0, 0.1)}, {0.1: _A2})

    namer.adopt_confirmed("pyannote:SPEAKER_00", "人物1")

    assert namer.confirmed_name("pyannote:SPEAKER_00") == "人物1"
    assert "pyannote:SPEAKER_00" not in namer._buffers
    assert namer.observe("pyannote:SPEAKER_00", _wav(1.0, 0.9)) == "人物1"


def test_link_threshold_matches_calibrated_constant():
    """下限は校正済み定数（0.50）そのもの＝ここを動かすなら再校正が要る."""
    from das.asr.live._constants import PYANNOTE_CLUSTER_MINT_LINK_MIN_SIM

    namer = _namer_with({"pyannote:SPEAKER_00": _wav(6.0, 0.1)}, {0.1: _MID})
    sim = float(np.dot(_A, _MID))
    assert sim > PYANNOTE_CLUSTER_MINT_LINK_MIN_SIM   # 0.6 > 0.50

    assert namer.link_minted_profile(_A, lambda raw: "@diar:1") is not None
    # 下限を実測値より上に動かすと同じペアが通らなくなる（校正の意味を固定）
    assert namer.link_minted_profile(_A, lambda raw: "@diar:1",
                                     min_sim=0.7) is None


# ---------------------------------------------------------------------------
# flush 配線: opt-in と台帳の書き換え
# ---------------------------------------------------------------------------

class _Args:
    lang = "ja"
    vp_debug = False
    vp_mint_cluster_link = False


class _Backend:
    def parse_message(self, raw, lang):
        return raw


class _MintTracker(_EmbedTracker):
    """flush の声紋前段で「自動登録」を1回返すフェイク."""

    def __init__(self, embed_map, profiles):
        super().__init__(embed_map, profiles)
        self.last = None

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        self.last = {"kind": "自動登録", "label": "1", "name": "人物1",
                     "rename": None, "chars": chars}
        return "人物1"

    def is_active_human(self, key):
        return key in self.profiles


class _VoiceResolver:
    """声紋の主張をそのまま採るフェイク SpeakerResolver."""

    def resolve(self, **kwargs):
        return SimpleNamespace(source="voiceprint",
                               speaker=kwargs["voiceprint_speaker"],
                               confidence=1.0, reason="voiceprint")


def _make_state(tmp_path, *, namer, tracker, link_enabled, max_speakers=2):
    state = SessionState(
        args=SimpleNamespace(diarization_max_speakers=max_speakers),
        started=datetime.datetime(2026, 7, 25),
        out_path=str(tmp_path / "o.md"), html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"), turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"), tracker=tracker, serve=False,
    )
    state.save = lambda *a, **k: None
    state.asr_pcm_buf = bytearray(b"\0" * SR * 2 * 5)
    state.cluster_namer = namer
    state.speaker_resolver = _VoiceResolver()
    args = _Args()
    args.vp_mint_cluster_link = link_enabled
    loop = RecvLoop(state, args, _Backend())
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000
    return state, loop


def _seated_setup(tmp_path, *, link_enabled):
    """@diar:1 が席を持ち、同一人物の声で 人物1 が鋳造される状況を作る."""
    tracker = _MintTracker({0.1: _A2}, {"人物1": _A})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer._buffers["pyannote:SPEAKER_00"] = [_wav(6.0, 0.1)]
    state, loop = _make_state(tmp_path, namer=namer, tracker=tracker,
                              link_enabled=link_enabled)
    # クラスタ側が先に席を取っている（二重帳簿の前提条件）
    state.key_for_diarization_speaker("pyannote", "SPEAKER_00")   # @diar:1
    state.records.append({"ms": 0, "end_ms": 500, "speaker": "@diar:1",
                          "text": "先に話していた人"})
    state.disp_name("@diar:1")
    return state, loop


def test_disabled_by_default_keeps_two_ledgers(tmp_path):
    """既定（opt-in 無効）では従来どおり別々の戸籍のまま＝挙動不変."""
    state, loop = _seated_setup(tmp_path, link_enabled=False)

    loop.flush()

    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_00": "@diar:1"}
    assert state.cluster_namer.confirmed_name("pyannote:SPEAKER_00") is None
    speakers = {r["speaker"] for r in state.records if "speaker" in r}
    assert speakers == {"@diar:1", "人物1"}   # 二重帳簿（現状の姿）


def test_enabled_merges_seat_into_minted_person(tmp_path):
    """有効時は席が人物1へ統合され、過去分・台帳・確定が揃って書き変わる."""
    state, loop = _seated_setup(tmp_path, link_enabled=True)

    loop.flush()

    # 過去の @diar:1 の発話も遡及的に 人物1 になる（rekey 経由）
    speakers = {r["speaker"] for r in state.records if "speaker" in r}
    assert speakers == {"人物1"}
    # 台帳: 以後この生クラスタの発話は 人物1 に解決される
    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_00": "人物1"}
    # クラスタ確定: observe が短絡で 人物1 を返す
    assert state.cluster_namer.confirmed_name("pyannote:SPEAKER_00") == "人物1"


def test_enabled_frees_the_seat_for_the_real_second_speaker(tmp_path):
    """統合で席が1つ空き、2人目（実在者）が締め出されなくなる（根治の本体）.

    二重帳簿のままだと上限2の会話で「同じ人が2席」を占め、2人目は未確定に
    落ち続ける（2026-07-25_1723 の実測。handoff §1）。
    """
    state, loop = _seated_setup(tmp_path, link_enabled=True)
    loop.flush()

    # 2人目のクラスタが席を要求する
    second = state.key_for_diarization_speaker("pyannote", "SPEAKER_01")
    assert state.constrain_human_speaker_key(second) == second   # 締め出されない


def test_enabled_but_no_match_leaves_ledgers_untouched(tmp_path):
    """別人の声なら統合しない（誤統合より二重帳簿のほうがまだ安全）."""
    tracker = _MintTracker({0.1: _B}, {"人物1": _A})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer._buffers["pyannote:SPEAKER_00"] = [_wav(6.0, 0.1)]
    state, loop = _make_state(tmp_path, namer=namer, tracker=tracker,
                              link_enabled=True)
    state.key_for_diarization_speaker("pyannote", "SPEAKER_00")

    loop.flush()

    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_00": "@diar:1"}
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None


def test_no_cluster_namer_is_unaffected(tmp_path):
    """cluster_namer が無い構成（Soniox単独/pyannote単独）は有効化しても不変."""
    tracker = _MintTracker({}, {"人物1": _A})
    state, loop = _make_state(tmp_path, namer=None, tracker=tracker,
                              link_enabled=True)

    loop.flush()   # 例外を出さずに従来どおり終わる

    assert state.records[-1]["speaker"] == "人物1"
    assert state.diarization_speaker_keys == {}


def test_link_is_recorded_in_diag(tmp_path):
    """統合は diag に構造化イベントとして残る（実地検証で観測可能にする）."""
    state, loop = _seated_setup(tmp_path, link_enabled=True)

    loop.flush()

    with open(state.diag_path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    link = next(e for e in events if e.get("type") == "mint_cluster_link")
    assert link["cluster"] == "pyannote:SPEAKER_00"
    assert link["seat"] == "@diar:1"
    assert link["name"] == "人物1"
    assert link["sim"] == round(float(np.dot(_A, _A2)), 3)
