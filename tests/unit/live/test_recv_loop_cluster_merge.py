"""flush配線のクラスタ間名寄せ（登録者ゼロ対策）の統合テスト.

設計: docs/design/handoff_2026-07-14_unregistered_speakers.md §3 参照。
cluster_namer 有効時のみ働く経路（名寄せ成立時の遡及統合、canonicalへの
キー集約、max-speakers超過時の最近傍統合）を、フェイクの namer/resolver で検証する。
"""
from __future__ import annotations

import datetime
from types import SimpleNamespace

from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState


class _Args:
    lang = "ja"
    vp_debug = False


class _Backend:
    def parse_message(self, raw, lang):
        return raw


class _Tracker:
    """flushの声紋前段を素通りさせる最小フェイク（kindは非高信頼の「蓄積中」）."""

    def __init__(self) -> None:
        self.last = {"kind": "蓄積中", "label": "1"}

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        return "#1"


class _Resolver:
    """常に外部diarization由来の話者を返すフェイク SpeakerResolver."""

    def __init__(self, speaker: str, source: str = "pyannote") -> None:
        self._speaker = speaker
        self._source = source

    def resolve(self, **kwargs):
        return SimpleNamespace(source=self._source, speaker=self._speaker,
                               confidence=0.9, reason="diarization_overlap")


class _Namer:
    """ClusterVoiceNamer の名寄せ結果だけを模したフェイク."""

    def __init__(self, aliases=None, nearest=None, nearest_sim=1.0,
                 dedupe=0.72) -> None:
        self._aliases = dict(aliases or {})
        self._nearest = nearest
        self._nearest_sim = nearest_sim
        # 最近傍統合の下限閾値は VoiceProfiles.dedupe を流用する（F2）。
        self.tracker = SimpleNamespace(dedupe=dedupe)
        self.last_match = None

    def observe(self, raw_cluster, wav, *, overlapped=False):
        return None   # 声紋名前付けは常に未確定（名寄せ経路の検証に集中する）

    def canonical_cluster(self, raw_cluster):
        return self._aliases.get(raw_cluster, raw_cluster)

    def nearest_cluster(self, raw_cluster, exclude=None):
        if self._nearest is None:
            return None
        return self._nearest, self._nearest_sim


class _FakePyannoteProvider:
    name = "pyannote"

    def drain_events(self):
        return []


def _make_state(tmp_path, *, namer, speaker, max_speakers=None):
    state = SessionState(  # type: ignore[no-untyped-call]
        args=SimpleNamespace(diarization_max_speakers=max_speakers),
        started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp_path / "o.md"),
        html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"),
        turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"),
        tracker=_Tracker(),
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.asr_pcm_buf = bytearray(b"\0" * 16000 * 2 * 5)
    state.cluster_namer = namer  # type: ignore[assignment]
    state.speaker_resolver = _Resolver(speaker)  # type: ignore[assignment]
    return state


def _flush(state, *, text="これは検証用の発言です", ms=1000, end=3000):
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = text
    loop.cur_ms = ms
    loop.cur_end = end
    loop.flush()  # type: ignore[no-untyped-call]
    return state


def test_merged_cluster_rekeys_absorbed_key_into_canonical(tmp_path):
    """名寄せ成立時、吸収側の @diar:N が rekey で canonical のキーへ遡及統合される."""
    namer = _Namer(aliases={"pyannote:SPEAKER_01": "pyannote:SPEAKER_00"})
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    state.diarization_speaker_keys = {"pyannote:SPEAKER_00": "@diar:1",
                                      "pyannote:SPEAKER_01": "@diar:2"}
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:2", "text": "昔の発言"}]

    _flush(state)

    assert state.records[0]["speaker"] == "@diar:1"    # 過去レコードの遡及統合
    assert state.records[-1]["speaker"] == "@diar:1"   # 新しい発話も canonical のキー
    assert "pyannote:SPEAKER_01" not in state.diarization_speaker_keys


def test_merged_cluster_without_canonical_key_reuses_absorbed_key(tmp_path):
    """canonical が未キーで吸収側にキーがある場合、そのキーを canonical へ付け替える."""
    namer = _Namer(aliases={"pyannote:SPEAKER_01": "pyannote:SPEAKER_00"})
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    state.diarization_speaker_keys = {"pyannote:SPEAKER_01": "@diar:1"}

    _flush(state)

    assert state.records[-1]["speaker"] == "@diar:1"
    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_00": "@diar:1"}


def test_merged_cluster_issues_key_under_canonical_when_both_unkeyed(tmp_path):
    """両方未キーなら canonical の source/speaker でキーを発行する（pending集約）."""
    namer = _Namer(aliases={"pyannote:SPEAKER_01": "pyannote:SPEAKER_00"})
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    # provider未設定＝ヒステリシスなし → 即時発行（canonical側で発行されることを確認）

    _flush(state)

    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_00": "@diar:1"}
    assert state.records[-1]["speaker"] == "@diar:1"


def test_unmerged_cluster_over_max_speakers_joins_nearest(tmp_path):
    """max-speakers到達後の未キークラスタは、最近傍クラスタの既存キーへ統合される."""
    namer = _Namer(nearest="pyannote:SPEAKER_00")
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_05", max_speakers=1)
    state.diarization_speaker_keys = {"pyannote:SPEAKER_00": "@diar:1"}
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "既存参加者"}]
    state.disp_name("@diar:1")   # 匿名ラベル（参加者A）を確定させ、スロット1を占有

    _flush(state)

    assert state.records[-1]["speaker"] == "@diar:1"   # 新規参加者を作らず統合
    assert "pyannote:SPEAKER_05" not in state.diarization_speaker_keys


def test_unmerged_cluster_over_max_speakers_without_nearest_falls_back(tmp_path):
    """最近傍が取れなければ従来どおり（constrainで未確定へ落ちる＝既存挙動）."""
    namer = _Namer(nearest=None)
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_05", max_speakers=1)
    state.key_for_diarization_speaker("pyannote", "SPEAKER_00")   # @diar:1
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "既存参加者"}]
    state.disp_name("@diar:1")

    _flush(state)

    assert state.records[-1]["speaker"] == "?"   # UNSURE_SPEAKER（上限超過の既存挙動）


def test_unmerged_cluster_over_max_speakers_below_dedupe_stays_unsure(tmp_path):
    """最近傍でも類似度が dedupe 未満なら統合せず、未確定に落とす（安全側）."""
    namer = _Namer(nearest="pyannote:SPEAKER_00", nearest_sim=0.3)
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_05", max_speakers=1)
    state.key_for_diarization_speaker("pyannote", "SPEAKER_00")   # @diar:1
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "既存参加者"}]
    state.disp_name("@diar:1")

    _flush(state)

    # 全く似ていない声は既存参加者に張り付かせず、従来経路（constrainで未確定）へ
    assert state.records[-1]["speaker"] == "?"


def test_unmerged_cluster_under_limit_issues_new_key(tmp_path):
    """名寄せ不成立でも上限未達なら従来どおり新規キーを発行する."""
    namer = _Namer()
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")

    _flush(state)

    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_01": "@diar:1"}
    assert state.records[-1]["speaker"] == "@diar:1"


def test_new_key_stays_unique_after_merge_shrinks_key_map(tmp_path):
    """名寄せの pop で keys が縮んでも、次の新規昇格キーは再利用されない（単調採番）."""
    namer = _Namer(aliases={"pyannote:SPEAKER_01": "pyannote:SPEAKER_00"})
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    state.key_for_diarization_speaker("pyannote", "SPEAKER_00")   # @diar:1
    state.key_for_diarization_speaker("pyannote", "SPEAKER_01")   # @diar:2

    _flush(state)   # 名寄せ成立: SPEAKER_01 が pop され @diar:2 → @diar:1 に遡及統合

    assert state.records[-1]["speaker"] == "@diar:1"
    state.speaker_resolver = _Resolver("SPEAKER_05")   # 別人の新クラスタが昇格
    _flush(state)
    # len ベース採番なら使用中の @diar:2 が再発行され別人が混在していた（F1回帰）。
    assert state.diarization_speaker_keys["pyannote:SPEAKER_05"] == "@diar:3"
    assert state.records[-1]["speaker"] == "@diar:3"


def test_merge_carries_absorbed_pending_into_canonical(tmp_path):
    """名寄せ成立時、吸収側のヒステリシス pending が canonical に合算される（F3）."""
    namer = _Namer()
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    state.diarization_provider = _FakePyannoteProvider()   # ヒステリシス有効(3s)

    _flush(state)   # 2秒発話: 閾値未満なので未確定のまま pending に蓄積
    assert state.records[-1]["speaker"] == "?"
    assert state.diarization_pending_ms == {"pyannote:SPEAKER_01": 2000}

    namer._aliases["pyannote:SPEAKER_01"] = "pyannote:SPEAKER_00"   # 名寄せ成立
    _flush(state)   # 2秒発話: 引き継いだ 2000ms と合算し 4000ms >= 3000ms で昇格

    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_00": "@diar:1"}
    assert state.records[-1]["speaker"] == "@diar:1"
    assert state.diarization_pending_ms == {}   # 吸収側の残留なし


def test_diag_records_final_key_after_constrain(tmp_path):
    """diag に constrain 後の最終キー(final_key)が追記される（既存フィールドは不変）.

    従来は constrain 前の key しか記録されず、「resolver は正しいキーを選んだのに
    constrain で未確定に落ちた」事象（2026-07-14 実セッション）の切り分けが
    diag からできなかった。final_key は追加のみで、diag 消費側の互換性を保つ。
    """
    namer = _Namer(nearest=None)
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_05", max_speakers=1)
    state.key_for_diarization_speaker("pyannote", "SPEAKER_00")   # @diar:1
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "既存参加者"}]
    state.disp_name("@diar:1")   # スロット1を占有 → 新キーは constrain で未確定へ

    _flush(state)

    import json
    with open(state.diag_path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if '"final_key"' in line]
    assert events, "final_key を含む diag 行が無い"
    ev = events[-1]
    assert ev["key"] == "@diar:2"     # constrain 前（resolver/名寄せの出力）は従来どおり
    assert ev["final_key"] == "?"     # constrain で未確定へ落ちたことが diag から読める
    assert state.records[-1]["speaker"] == ev["final_key"]   # records と一致


def test_cluster_namer_last_match_written_to_diag_once(tmp_path):
    """名寄せイベントが diag に1行書かれ、消費されて重複出力しない（F6）."""
    namer = _Namer()
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    namer.last_match = {"kind": "クラスタ名寄せ", "raw": "pyannote:SPEAKER_01",
                        "canonical": "pyannote:SPEAKER_00", "sim": 0.9}

    _flush(state)
    _flush(state)   # 2回目は last_match が消費済みなので書かれない

    import json
    with open(state.diag_path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f
                  if '"cluster_naming"' in line]
    assert len(events) == 1
    assert events[0]["type"] == "cluster_naming"
    assert events[0]["kind"] == "クラスタ名寄せ"
    assert events[0]["canonical"] == "pyannote:SPEAKER_00"
    assert namer.last_match is None
