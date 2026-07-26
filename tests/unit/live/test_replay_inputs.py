"""判定の入力が記録され、記録から本番コードを再生できることのテスト.

背景（handoff §23）: クラスタ層の入力（diarization provider の話者区間）は
メモリ上にしか無く実行後に失われていたため、記録から本番の判定経路を再生
できなかった。eval/replay_attribution.py が「クラスタ帰属は再現不可」として
声紋層しか再生できなかったのはこのため。設計変更のたびに近似ハーネスを書くか、
揺れの大きいライブ1ラン比較に頼るしかなかった。

ここで固定するのは2点:
  - 記録側: diag に判定の入力（diar 窓・classify フラグ・構成）が残ること
  - 再生側: その記録を本番コード（decide_speaker / SessionState）に流すと
    記録どおりの結論が出ること（＝再生が忠実であること）
"""
from __future__ import annotations

import datetime
import json
import sys
import wave
from pathlib import Path

import numpy as np

from das.asr.live._constants import SR, UNSURE_SPEAKER
from das.asr.live._diarization import DiarizationEvent, SpeakerResolver
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))


# ---------------------------------------------------------------------------
# 記録側
# ---------------------------------------------------------------------------

class _Args:
    lang = "ja"
    vp_debug = False
    vp_mint_cluster_link = False
    diarization = "pyannote"
    diarization_max_speakers = 2
    vp_cluster_naming = True
    stt = "soniox"


class _Tracker:
    """flush の声紋段を素通りさせる最小フェイク."""

    model = "redimnet"
    auto = True
    hybrid = True

    def __init__(self) -> None:
        self.last = {"kind": "蓄積中", "label": "1"}
        self.calls: list[dict] = []

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        self.calls.append({"overlapped": overlapped, "count": count,
                           "enroll": enroll, "chars": chars})
        return "#1"


class _Provider:
    name = "pyannote"

    def drain_events(self):
        return []

    def active_events(self):
        return []


def _state_for_recording(tmp_path, events):
    state = SessionState(
        args=_Args(), started=datetime.datetime(2026, 7, 25),
        out_path=str(tmp_path / "o.md"), html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"), turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"), tracker=_Tracker(), serve=False,
        diarization_provider=_Provider(), speaker_resolver=SpeakerResolver())
    state.save = lambda *a, **k: None
    state.asr_pcm_buf = bytearray(b"\0" * SR * 2 * 10)
    state.diarization_window = lambda start_ms, end_ms: list(events)
    return state


def _diag_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_diag_records_the_diarization_window_seen_by_the_decision(tmp_path):
    """判定が見た話者区間の窓が diag に残る（クラスタ層の入力の保存）."""
    events = [DiarizationEvent(start_ms=900, end_ms=3100,
                               speaker="SPEAKER_00", source="pyannote")]
    state = _state_for_recording(tmp_path, events)
    loop = RecvLoop(state, _Args(), backend=None)
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000

    loop.flush()

    utt = next(d for d in _diag_lines(state.diag_path) if "label" in d)
    assert utt["diar"] == [["pyannote", "SPEAKER_00", 900, 3100]]


def test_diag_records_the_classify_flags(tmp_path):
    """classify に実際に渡した条件が残る（enroll は記録からは再現できないため）."""
    state = _state_for_recording(tmp_path, [])
    loop = RecvLoop(state, _Args(), backend=None)
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000

    loop.flush()

    utt = next(d for d in _diag_lines(state.diag_path) if "label" in d)
    assert utt["ov"] is False
    assert utt["enr"] is True
    assert utt["chars"] == len("これは検証用の発言です")
    # 記録した値は classify に渡した値と一致する
    assert state.tracker.calls[-1]["enroll"] is True


def test_diag_records_which_path_decided_the_speaker(tmp_path):
    """帰属がどの経路で決まったかが diag に残る（handoff §26.6）.

    speaker_source は records にしか無く、records は終了時に永続化されない
    （transcripts に残るのは diag/turns/wav だけ）。そのためオフライン分析が
    「この誤帰属は 3d の門番で止められる経路か、STT フォールバックか」を
    分けられず、門番の適用範囲を数字で決められなかった。
    """
    state = _state_for_recording(tmp_path, [])   # 窓が空＝STT フォールバック
    loop = RecvLoop(state, _Args(), backend=None)
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000

    loop.flush()

    utt = next(d for d in _diag_lines(state.diag_path) if "label" in d)
    assert utt["src"] == "stt_fallback"
    assert utt["why"] == state.records[-1]["speaker_reason"]
    # diag の経路は records の経路と同じもの（二重管理にしない）
    assert utt["src"] == state.records[-1]["speaker_source"]


def test_diag_does_not_leak_decision_inputs_into_records(tmp_path):
    """判定の入力は diag 限定で、議事録レコードには混ぜない（出力形式の互換）."""
    events = [DiarizationEvent(start_ms=900, end_ms=3100,
                               speaker="SPEAKER_00", source="pyannote")]
    state = _state_for_recording(tmp_path, events)
    loop = RecvLoop(state, _Args(), backend=None)
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000

    loop.flush()

    rec = state.records[-1]
    assert "diar" not in rec and "ov" not in rec and "enr" not in rec


def test_session_config_is_written_to_diag(tmp_path):
    """構成が diag の1行として残る（どの設定で録れたランかを後から確定できる）."""
    from das.asr.live._bootstrap import write_session_config

    state = _state_for_recording(tmp_path, [])
    write_session_config(state, _Args(), state.tracker)

    cfg = next(d for d in _diag_lines(state.diag_path)
               if d.get("type") == "session_config")
    assert cfg["diarization"] == "pyannote"
    assert cfg["diarization_max_speakers"] == 2
    assert cfg["vp_cluster_naming"] is True
    assert cfg["vp_mint_cluster_link"] is False
    assert cfg["vp_model"] == "redimnet"


# ---------------------------------------------------------------------------
# 再生側（記録 → 本番コード → 同じ結論が出るか）
# ---------------------------------------------------------------------------

class _ScriptedTracker:
    """発話ごとに決められた判定を返すフェイク（声紋モデルを読まないため）."""

    model = "redimnet"
    auto = True
    hybrid = True

    def __init__(self, script) -> None:
        self._script = list(script)
        self.last = None
        self.profiles: dict = {}
        self.max_human_speakers = None

    def set_max_human_speakers(self, v):
        self.max_human_speakers = v

    def set_hybrid(self, v):
        self.hybrid = bool(v)

    def is_active_human(self, key):
        return key in self.profiles

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        key, last = self._script.pop(0)
        self.last = last
        return key


def _write_recording(root: Path, session: str, utts, texts, *, config):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"{session}.diag.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(config, ensure_ascii=False) + "\n")
        for u in utts:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    with open(root / f"{session}.turns.jsonl", "w", encoding="utf-8") as f:
        for i, (ms, text) in enumerate(texts, 1):
            f.write(json.dumps({"turn_id": i, "ms": ms, "end_ms": ms + 2000,
                                "speaker": "未確定", "text": text},
                               ensure_ascii=False) + "\n")
    with wave.open(str(root / f"{session}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.zeros(SR * 20, dtype="<i2").tobytes())


def test_replay_reproduces_the_recorded_decisions(tmp_path):
    """記録を本番コードに流すと、記録どおりの最終キーが再現される.

    これが成り立つ＝クラスタ層を含む帰属の全経路がオフラインで回せる、という
    このセッションの目的そのもの。ヒステリシス（累積3秒未満は未確定）と
    確定後の安定も含めて再現されることを見る。
    """
    import replay_live_attribution as rla

    utts = [
        # 1発話目: 累積2秒 → ヒステリシス未達で未確定
        {"ms": 1000, "end": 3000, "label": "1", "key": "?", "final_key": "?",
         "kind": "蓄積中", "ov": False, "enr": True, "chars": 10,
         "diar": [["pyannote", "SPEAKER_00", 900, 3100]]},
        # 2発話目: 累積4秒 → @diar:1 を発行
        {"ms": 4000, "end": 6000, "label": "1", "key": "@diar:1",
         "final_key": "@diar:1", "kind": "蓄積中", "ov": False, "enr": True,
         "chars": 10, "diar": [["pyannote", "SPEAKER_00", 3900, 6100]]},
        # 3発話目: 同じクラスタは同じキーで安定
        {"ms": 7000, "end": 9000, "label": "1", "key": "@diar:1",
         "final_key": "@diar:1", "kind": "蓄積中", "ov": False, "enr": True,
         "chars": 10, "diar": [["pyannote", "SPEAKER_00", 6900, 9100]]},
    ]
    texts = [(1000, "最初の発言です"), (4000, "次の発言です"), (7000, "三つ目の発言です")]
    config = {"type": "session_config", "diarization": "pyannote",
              "diarization_max_speakers": 2, "vp_cluster_naming": False,
              "vp_mint_cluster_link": False, "vp_model": "redimnet",
              "vp_auto": True, "vp_hybrid": True, "stt": "soniox"}
    _write_recording(tmp_path, "s", utts, texts, config=config)

    rec = rla.load_session("s", root=tmp_path)
    assert rla.has_replayable_inputs(rec["utts"])
    tracker = _ScriptedTracker([("#1", {"kind": "蓄積中", "label": "1"})] * 3)
    rows = rla.replay(rec, {}, tmp_path / "work", tracker=tracker)

    assert [r["pred"] for r in rows] == ["?", "@diar:1", "@diar:1"]
    fid, n = rla.fidelity(rows)
    assert (fid, n) == (1.0, 3)   # 記録と完全一致＝再生が忠実


def test_replay_can_change_configuration_to_compare_designs(tmp_path):
    """構成を上書きして同じ記録を流し直せる（設計比較の土台）."""
    import replay_live_attribution as rla

    utts = [
        {"ms": 1000, "end": 5000, "label": "1", "key": "@diar:1",
         "final_key": "@diar:1", "kind": "蓄積中", "ov": False, "enr": True,
         "chars": 10, "diar": [["pyannote", "SPEAKER_00", 900, 5100]]},
        {"ms": 6000, "end": 10000, "label": "2", "key": "@diar:2",
         "final_key": "@diar:2", "kind": "蓄積中", "ov": False, "enr": True,
         "chars": 10, "diar": [["pyannote", "SPEAKER_01", 5900, 10100]]},
    ]
    texts = [(1000, "一人目の発言"), (6000, "二人目の発言")]
    config = {"type": "session_config", "diarization": "pyannote",
              "diarization_max_speakers": 2, "vp_cluster_naming": False,
              "vp_mint_cluster_link": False, "vp_model": "redimnet",
              "vp_auto": True, "vp_hybrid": True, "stt": "soniox"}
    _write_recording(tmp_path, "s", utts, texts, config=config)
    rec = rla.load_session("s", root=tmp_path)

    def _run(overrides, work):
        tracker = _ScriptedTracker([("#1", {"kind": "蓄積中", "label": "1"}),
                                    ("#2", {"kind": "蓄積中", "label": "2"})])
        return [r["pred"] for r in rla.replay(rec, overrides, work, tracker=tracker)]

    assert _run({}, tmp_path / "a") == ["@diar:1", "@diar:2"]
    # 想定話者数を1に絞ると2人目が席を得られない（constrain が効く）
    assert _run({"diarization_max_speakers": 1}, tmp_path / "b") == ["@diar:1", "?"]


def test_old_recordings_are_reported_as_not_replayable(tmp_path):
    """入力が記録されていない旧ランは、黙って近似せず「再生できない」と分かる."""
    import replay_live_attribution as rla

    utts = [{"ms": 1000, "end": 3000, "label": "1", "key": "?",
             "final_key": "?", "kind": "蓄積中"}]
    _write_recording(tmp_path, "old", utts, [(1000, "旧ランの発言")],
                     config={"type": "session_config"})

    rec = rla.load_session("old", root=tmp_path)
    assert rla.has_replayable_inputs(rec["utts"]) is False


# ---------------------------------------------------------------------------
# 録音後の診断（eval/diagnose_live_session.py）
# ---------------------------------------------------------------------------

def test_diagnosis_measures_cluster_fragmentation(tmp_path):
    """クラスタ分裂を「発話時間の90%を覆うクラスタ数」で測る（項目2の判断材料）.

    観測クラスタ数をそのまま使うと、1発話だけの雑音クラスタが数を膨らませて
    分裂の深刻さを過大評価する。時間で重み付けした実質クラスタ数を見る。
    """
    import diagnose_live_session as diag

    # 2人の会話に、主要2クラスタ＋雑音3クラスタ（各1発話・短い）
    utts = []
    for i in range(10):
        ms = 1000 + i * 5000
        spk = "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01"
        utts.append({"ms": ms, "end": ms + 4000, "label": "1",
                     "key": "@diar:1", "final_key": "@diar:1", "kind": "蓄積中",
                     "ov": False, "enr": True, "chars": 10,
                     "diar": [["pyannote", spk, ms, ms + 4000]]})
    for j, spk in enumerate(("SPEAKER_07", "SPEAKER_08", "SPEAKER_09")):
        ms = 60000 + j * 1000
        utts.append({"ms": ms, "end": ms + 200, "label": "1",
                     "key": "?", "final_key": "?", "kind": "照合なし",
                     "ov": False, "enr": True, "chars": 2,
                     "diar": [["pyannote", spk, ms, ms + 200]]})
    texts = [(u["ms"], "検証用の発言です") for u in utts]
    _write_recording(tmp_path, "s", utts, texts,
                     config={"type": "session_config", "diarization": "pyannote",
                             "diarization_max_speakers": 2})

    rec = diag.read_diag("s", root=tmp_path)
    sec: dict[str, float] = {}
    for u in rec["utts"]:
        dur = (u["end"] - u["ms"]) / 1000.0
        for src, spk, _s, _e in u.get("diar", []):
            sec[f"{src}:{spk}"] = sec.get(f"{src}:{spk}", 0.0) + dur
    total = sum(sec.values())
    acc, effective = 0.0, 0
    for _k, v in sorted(sec.items(), key=lambda kv: -kv[1]):
        acc += v
        effective += 1
        if acc >= total * 0.9:
            break
    assert len(sec) == 5        # 観測は5クラスタ
    assert effective == 2       # 実質は2＝想定話者数と一致（分裂していない）


def test_diagnosis_marks_old_recordings_as_a_lower_bound(tmp_path, capsys):
    """旧ランでは再生不可を明示し、分裂は「下限」として測る（黙って断定しない）.

    旧ランで見えるのは「ヒステリシスを超え、かつ少なくとも1発話を取った
    クラスタ」だけ。声紋が勝った発話のクラスタ所属と、キーを得る前に消えた
    クラスタは見えないので、真の分裂はこれ以上にしかならない。下限と明示した
    うえで出す（下限が既に大きければ結論は動かないため、使い道はある）。
    """
    import diagnose_live_session as diag

    utts = [
        {"ms": 1000, "end": 5000, "label": "1", "key": "@diar:1",
         "final_key": "@diar:1", "kind": "蓄積中"},
        {"ms": 6000, "end": 10000, "label": "2", "key": "@diar:2",
         "final_key": "@diar:2", "kind": "蓄積中"},
        # 声紋が勝った発話はクラスタ所属が見えない（下限になる理由）
        {"ms": 11000, "end": 15000, "label": "1", "key": "人物1",
         "final_key": "人物1", "kind": "声紋一致"},
    ]
    _write_recording(tmp_path, "old", utts,
                     [(u["ms"], "旧ランの発言") for u in utts],
                     config={"type": "session_config"})

    rec = diag.read_diag("old", root=tmp_path)
    diag.report_fidelity("old", rec)
    diag.report_fragmentation(rec)
    out = capsys.readouterr().out
    assert "判定の入力が記録されていない" in out
    assert "下限" in out
    assert "観測クラスタ数（下限）: 2" in out   # 人物1 の発話は数えられない


# ---------------------------------------------------------------------------
# 文字起こし精度（eval/score_transcription.py）
# ---------------------------------------------------------------------------

def test_transcription_normalization_strips_transcript_markup():
    """転記記号を落とし、実際に話された語は残す（Chiba は CSJ 系の転記規則）."""
    import score_transcription as sc

    src = "(F_あのね:)(D_ア)塾講の:ことなんだけど:<笑>(1.547)中二の子"
    assert sc.normalize(src) == "あのねア塾講のことなんだけど中二の子"
    # フィラー・言い直しを除く指定では中身ごと落ちる
    assert sc.normalize(src, keep_filler=False) == "塾講のことなんだけど中二の子"


def test_transcription_normalization_levels_punctuation_and_width():
    """句読点・空白・全角半角の差は誤りに数えない（土俵を揃える）."""
    import score_transcription as sc

    assert sc.normalize("で、恋の話なんですけど。") == sc.normalize("で恋の話なんですけど")
    assert sc.normalize("ＡＢＣ") == sc.normalize("ABC")


def test_cer_counts_edits_against_reference_length():
    """CER は編集距離 ÷ 参照長（挿入・削除・置換を同じ重みで数える）."""
    import score_transcription as sc

    assert sc.cer("あいうえお", "あいうえお") == (0, 5)
    assert sc.cer("あいうえお", "あいうXお") == (1, 5)     # 置換1
    assert sc.cer("あいうえお", "あいうえおか") == (1, 5)   # 挿入1
    assert sc.cer("あいうえお", "あいうえ") == (1, 5)       # 削除1


def test_full_concatenation_is_the_default_alignment():
    """既定は全文連結（時間窓で仕切ると区切りの違いが誤りに化けるため）.

    当初は「参照は相槌を独立ターンで挟むのに対しシステムは長くまとめるので
    全文連結だと順序ずれが誤差になる」と考えて10秒窓を試したが、実測で
    CER は 32%→50% と悪化した。システムのターンが40秒級で複数の窓をまたぎ、
    参照だけが後続の窓に残って全削除に数えられるため。既定を戻した経緯を
    ここで固定する（同じ「改善」を繰り返さないため）。
    """
    import inspect

    import score_transcription as sc
    sig = inspect.signature(sc.score_run)
    assert sig.parameters["window_sec"].default == 0.0


# ---------------------------------------------------------------------------
# 席落ちの割当て（handoff §27）: flush の呼び出し口
# ---------------------------------------------------------------------------

def _loop_with_seat_audio(tmp_path, *, seats):
    """席上限で落ちる状況を作り、SeatAudio を差した RecvLoop を返す."""
    from das.asr.live._seat_audio import SeatAudio
    state = _state_for_recording(tmp_path, [])
    state.seat_audio = SeatAudio(_SeatTracker(), ref_sec=30.0, min_ref_sec=1.0)
    for key, tag in seats:
        state.seat_audio.observe(key, np.full(SR * 2, tag, dtype=np.float32))
    return state, RecvLoop(state, _Args(), backend=None)


class _SeatTracker:
    """先頭サンプルの符号で人物を分ける最小の埋め込み器."""

    def embed_audio(self, wav):
        if wav is None or wav.size == 0:
            return None
        v = np.array([1.0, 0.0] if float(wav[0]) >= 0 else [0.0, 1.0])
        return v


def test_seat_drop_is_assigned_to_the_nearest_seat(tmp_path, monkeypatch):
    """上流が決めていたのに席上限で落ちた発話は、席の実音声で寄せ直される."""
    state, loop = _loop_with_seat_audio(
        tmp_path, seats=[("人物1", 1.0), ("人物2", -1.0)])
    # 上流は決めている / constrain が未確定に落とす、という状況を作る
    monkeypatch.setattr("das.asr.live._recv_loop.decide_speaker",
                        lambda *a, **k: "@diar:3")
    monkeypatch.setattr(state, "constrain_human_speaker_key",
                        lambda k: UNSURE_SPEAKER)
    state.asr_pcm_buf = bytearray(np.full(SR * 10, 20000, dtype="<i2").tobytes())
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000

    loop.flush()

    rec = state.records[-1]
    assert rec["speaker"] == "人物1"          # 未確定ではなく席へ寄った
    assert rec["speaker_source"] == "seat_assign"
    assert rec["speaker_reason"] == "seat_full_nearest_seat_audio"


def test_seat_assignment_does_not_fire_when_upstream_is_unsure(tmp_path,
                                                               monkeypatch):
    """上流が決めていない発話には掛けない（席の問題ではないため）."""
    state, loop = _loop_with_seat_audio(
        tmp_path, seats=[("人物1", 1.0), ("人物2", -1.0)])
    monkeypatch.setattr(state, "constrain_human_speaker_key",
                        lambda k: UNSURE_SPEAKER)
    monkeypatch.setattr("das.asr.live._recv_loop.decide_speaker",
                        lambda *a, **k: UNSURE_SPEAKER)
    state.asr_pcm_buf = bytearray(np.full(SR * 10, 20000, dtype="<i2").tobytes())
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000

    loop.flush()

    assert state.records[-1]["speaker"] == UNSURE_SPEAKER


def test_seat_assignment_writes_no_confirmation(tmp_path, monkeypatch):
    """割当ては1発話限りで、次の発話は独立に判定される（可逆性の担保）.

    §15.12 の一般則「不可逆な操作は高確信を要求する」との整合はここで取る。
    確定を書かないからこそ、類似度の下限を課さずに済む。
    """
    state, loop = _loop_with_seat_audio(
        tmp_path, seats=[("人物1", 1.0), ("人物2", -1.0)])
    monkeypatch.setattr("das.asr.live._recv_loop.decide_speaker",
                        lambda *a, **k: "@diar:3")
    monkeypatch.setattr(state, "constrain_human_speaker_key",
                        lambda k: UNSURE_SPEAKER)
    state.asr_pcm_buf = bytearray(np.full(SR * 10, 20000, dtype="<i2").tobytes())
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000
    loop.flush()
    assert state.records[-1]["speaker"] == "人物1"

    # 次の発話は逆の声 → 台帳に引きずられず、独立に人物2へ寄る
    state.asr_pcm_buf = bytearray(np.full(SR * 10, -20000, dtype="<i2").tobytes())
    loop.cur_speaker = "1"
    loop.cur_text = "こちらは別の人の発言です"
    loop.cur_ms, loop.cur_end = 4000, 6000
    loop.flush()

    assert state.records[-1]["speaker"] == "人物2"
    assert state.diarization_speaker_keys == {}   # 台帳に書いていない


def test_no_seat_audio_means_unchanged_behaviour(tmp_path, monkeypatch):
    """seat_audio が無い構成（pyannote単独・Soniox単独）は完全に不変."""
    state = _state_for_recording(tmp_path, [])
    assert state.seat_audio is None
    loop = RecvLoop(state, _Args(), backend=None)
    monkeypatch.setattr("das.asr.live._recv_loop.decide_speaker",
                        lambda *a, **k: "@diar:3")
    monkeypatch.setattr(state, "constrain_human_speaker_key",
                        lambda k: UNSURE_SPEAKER)
    loop.cur_speaker = "1"
    loop.cur_text = "これは検証用の発言です"
    loop.cur_ms, loop.cur_end = 1000, 3000

    loop.flush()

    assert state.records[-1]["speaker"] == UNSURE_SPEAKER
