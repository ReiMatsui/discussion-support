"""scripts/enroll_voices.py と --activate（事前登録の再生評価の経路）のテスト."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from das.asr.live import _bootstrap
from das.asr.live._constants import SR
from das.asr.live._voice_profiles import VoiceProfiles

ROOT = Path(__file__).resolve().parents[3]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "enroll_voices", ROOT / "scripts" / "enroll_voices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_add_accepts_name_path_and_optional_range() -> None:
    ev = _load_script()
    assert ev.parse_add("田中=a.wav") == ("田中", "a.wav", None, None)
    assert ev.parse_add("田中=a.wav@30-90") == ("田中", "a.wav", 30.0, 90.0)
    assert ev.parse_add("田中=a.wav@-45") == ("田中", "a.wav", None, 45.0)


def test_build_clip_skips_short_segments_and_stops_at_requested_seconds() -> None:
    """相槌級の短い区間は使わず、要求秒数に達したところで止まる."""
    ev = _load_script()
    wav = np.arange(SR * 10, dtype="float32")   # 10秒。値=サンプル位置
    segs = [(0.0, 0.3), (1.0, 3.0), (4.0, 5.5), (6.0, 9.0)]
    clip, until = ev.build_clip(wav, segs, seconds=3.0)
    # 0.3秒は捨て、1-3(2s)+4-5.5(1.5s)=3.5s で 3s に達したので 6-9 は使わない
    assert abs(clip.size / SR - 3.5) < 1e-6
    assert until == 5.5
    assert clip[0] == SR * 1.0, "最初の区間は 1.0 秒から始まる"


def test_build_clip_after_uses_only_segments_starting_later() -> None:
    """after より前に始まる区間は使わない（採点する先頭と登録音声を重ねない）."""
    ev = _load_script()
    wav = np.arange(SR * 10, dtype="float32")
    segs = [(1.0, 3.0), (4.0, 5.5), (6.0, 9.0)]
    clip, until = ev.build_clip(wav, segs, seconds=2.0, after=5.0)
    assert clip[0] == SR * 6.0 and until == 9.0


def test_build_clip_returns_empty_when_nothing_usable() -> None:
    ev = _load_script()
    clip, until = ev.build_clip(np.zeros(SR, dtype="float32"), [(0.0, 0.2)], 5.0)
    assert clip.size == 0 and until == 0.0


def test_enroll_from_audio_persists_named_profile(tmp_path) -> None:
    """enroll_from_audio は名前付きで voices.json に残る（登録の経路そのもの）."""
    path = tmp_path / "voices.json"
    vp = VoiceProfiles(path=str(path), model="redimnet", auto=False,
                       embedder=lambda w: np.array([1.0, 0.0]))
    assert vp.enroll_from_audio("A", np.zeros(SR * 5, dtype="float32"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "A" in saved and saved["_model"] == "redimnet"
    assert vp.all_profile_names() == ["A"]


def test_activate_option_turns_on_saved_profiles_at_startup(capsys) -> None:
    """--activate は voices.json の登録済み声紋を起動時に照合対象へ入れる."""
    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp.profiles = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    vp._active_keys = set()
    vp.path = "voices.json"
    vp.dedupe = 0.9
    vp.own_sims = {}
    vp.own_embs = {}
    import threading
    vp._lock = threading.RLock()

    done = _bootstrap._activate_saved_profiles(vp, "A, C")
    assert done == ["A"]
    assert vp._active_keys == {"A"}
    assert "「C」" in capsys.readouterr().out, "無い名前は警告する"

    assert _bootstrap._activate_saved_profiles(vp, "all") == ["A", "B"]
    assert vp._active_keys == {"A", "B"}
    assert _bootstrap._activate_saved_profiles(vp, "") == []


def test_live_args_default_activate_is_empty() -> None:
    assert _bootstrap.LiveArgs().activate == ""


def test_check_level_flags_silent_short_and_clipped_recordings() -> None:
    ev = _load_script()
    assert "短すぎ" in ev.check_level(np.zeros(SR, dtype="float32"))
    assert "無音" in ev.check_level(np.zeros(SR * 5, dtype="float32"))
    loud = np.ones(SR * 5, dtype="float32")
    assert "クリップ" in ev.check_level(loud)
    ok = (np.sin(np.arange(SR * 5) / 20.0) * 0.1).astype("float32")
    assert ev.check_level(ok) is None


def test_save_wav_16k_roundtrips(tmp_path) -> None:
    ev = _load_script()
    wav = (np.sin(np.arange(SR * 3) / 30.0) * 0.5).astype("float32")
    out = tmp_path / "v" / "田中.wav"
    ev.save_wav_16k(out, wav)
    back = ev.read_wav_16k(out)
    assert back.size == wav.size and float(np.abs(back - wav).max()) < 1e-3


def test_preflight_voices_check_reports_missing_names(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts" / "preflight.py")
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({"_model": "redimnet", "田中": [1.0, 0.0], "人物1": [0.0, 1.0]}),
                    encoding="utf-8")
    res = pf.check_voices(str(path), "田中,佐藤")
    msgs = " / ".join(m for _, m in res)
    assert "田中" in msgs and "登録が無い参加者: 佐藤" in msgs
    assert any(mark == pf.NG for mark, _ in res)
    assert pf.check_voices(None, None) == []
    assert pf.check_voices(str(tmp_path / "none.json"), None)[0][0] == pf.NG


# ----------------------------------------------------------------------
# ゼミ録音など、過去の収録セッションから登録して別セッションを流す経路
# ----------------------------------------------------------------------
def test_parse_from_session_splits_name_session_label_and_range() -> None:
    ev = _load_script()
    assert ev.parse_from_session("伊藤先生=2026-06-25_140652/伊藤先生") == (
        "伊藤先生", "2026-06-25_140652", "伊藤先生", None, None)
    assert ev.parse_from_session("岡田さん=2026-06-25_1520/岡田さん@600-") == (
        "岡田さん", "2026-06-25_1520", "岡田さん", 600.0, None)
    assert ev.parse_from_session("A=s/ラベル@30-90")[3:] == (30.0, 90.0)


def test_session_segments_picks_only_that_label_and_merges_adjacent(tmp_path) -> None:
    ev = _load_script()
    rows = [
        {"turn_id": 1, "ms": 0, "end_ms": 1000, "speaker": "A", "text": "a"},
        {"turn_id": 2, "ms": 1200, "end_ms": 3000, "speaker": "A", "text": "b"},   # 0.2s 空き → 繋ぐ
        {"turn_id": 3, "ms": 3100, "end_ms": 5000, "speaker": "B", "text": "c"},
        {"turn_id": 4, "ms": 8000, "end_ms": 9000, "speaker": "A", "text": "d"},
        {"turn_id": 5, "ms": 9500, "end_ms": None, "speaker": "A", "text": "e"},  # 終端なしは捨てる
    ]
    (tmp_path / "s.turns.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    assert ev.session_segments("s", "A", transcripts=tmp_path) == [(0.0, 3.0), (8.0, 9.0)]
    assert ev.session_segments("s", "C", transcripts=tmp_path) == []


def test_gt_timeline_accepts_more_than_three_speakers() -> None:
    sys.path.insert(0, str(ROOT / "eval"))
    import _gtlib
    turns = [{"turn_id": i, "ms": i * 1000, "end_ms": i * 1000 + 900} for i in range(6)]
    labels = {"0": "S1", "1": "S4", "2": "S9", "3": "MULTI", "4": "UNK", "5": "S2"}
    tl = _gtlib.gt_timeline(turns, labels)
    assert set(tl) == {"S1", "S2", "S4", "S9"}
    assert _gtlib.is_speaker_code("S10") and not _gtlib.is_speaker_code("MULTI")


def test_best_assignment_is_one_to_one_and_maximises_chars() -> None:
    sys.path.insert(0, str(ROOT / "eval"))
    import enroll_breakdown as eb
    from collections import Counter
    cnt = {"a": Counter({"S1": 10, "S2": 3}), "b": Counter({"S1": 9}),
           "c": Counter({"S3": 5}), "d": Counter()}
    # a→S1 (10) + c→S3 (5) = 15 より、a→S2 (3) + b→S1 (9) + c→S3 (5) = 17 が大きい
    assert eb.best_assignment(cnt, ["S1", "S2", "S3", "S4", "S5"]) == {"a": "S2", "b": "S1", "c": "S3"}
    assert eb.best_assignment({"x": Counter()}, ["S1"]) == {}


def test_run_pair_builds_two_sessions_and_cuts_head(tmp_path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("run_pair", ROOT / "eval" / "run_pair.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    monkeypatch.setattr(rp, "PAIRS", tmp_path / "pairs")
    src = tmp_path / "in.wav"
    _load_script().save_wav_16k(src, np.zeros(SR * 90, dtype="float32"))
    cut = rp.head_wav(src, 1.0, "zemi")
    assert cut.name == "zemi_m1.wav" and _load_script().read_wav_16k(cut).size == SR * 60

    jobs = rp.build_commands(cut, "zemi", Path("v.json"), 5)
    assert [s for s, _ in jobs] == ["zemi_none", "zemi_enroll"]
    none_cmd, enroll_cmd = (" ".join(c) for _, c in jobs)
    assert "--no-intervention" in none_cmd and "--max-speakers 5" in none_cmd
    assert "--voices" not in none_cmd and "zemi_none.md" in none_cmd
    assert "--voices v.json --activate all" in enroll_cmd and "zemi_enroll.md" in enroll_cmd
    assert len(rp.build_commands(cut, "zemi", None, 3)) == 1, "--voices 無しは登録なしのみ"


def test_same_name_given_twice_is_concatenated_before_enrolling(tmp_path, monkeypatch) -> None:
    """--add 名前=…@a-b を複数回渡したら繋いで 1 人ぶんにする（上書きで最後だけにならない）."""
    ev = _load_script()
    src = tmp_path / "s.wav"
    ev.save_wav_16k(src, (np.sin(np.arange(SR * 20) / 25.0) * 0.3).astype("float32"))
    seen: dict[str, int] = {}

    class _VP:
        def __init__(self, path, auto=False):
            self.path = path

        def enroll_from_audio(self, name, wav):
            seen[name] = wav.size
            return True

        def all_profile_names(self):
            return list(seen)

    import types
    fake = types.ModuleType("das.asr.live._voice_profiles")
    fake.VoiceProfiles = _VP
    monkeypatch.setitem(sys.modules, "das.asr.live._voice_profiles", fake)
    ev.main(["--voices", str(tmp_path / "v.json"),
             "--add", f"A={src}@0-3", "--add", f"A={src}@10-14", "--add", f"B={src}@5-7"])
    assert seen == {"A": SR * 7, "B": SR * 2}


def test_from_gt_uses_hand_labelled_segments_from_turns_or_vad(tmp_path) -> None:
    """耳で付けた正解（annotate.py の labels）から区間を引く。turns 型と自動区切り型の両方."""
    ev = _load_script()
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "eval").mkdir()
    rows = [{"turn_id": 1, "ms": 0, "end_ms": 2000}, {"turn_id": 2, "ms": 2100, "end_ms": 4000},
            {"turn_id": 3, "ms": 5000, "end_ms": 7000}, {"turn_id": 4, "ms": 8000, "end_ms": 9000}]
    (tmp_path / "transcripts" / "s.turns.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    ev.save_wav_16k(tmp_path / "transcripts" / "s.wav", np.zeros(SR * 10, dtype="float32"))
    gt = tmp_path / "eval" / "gt_s.json"
    gt.write_text(json.dumps({"session": "s", "labels": {"1": "S1", "2": "S1", "3": "MULTI", "4": "S2"}}),
                  encoding="utf-8")
    segs, wav = ev.gt_segments(gt, "S1", root=tmp_path)
    assert segs == [(0.0, 4.0)] and wav == tmp_path / "transcripts" / "s.wav"
    assert ev.gt_segments(gt, "S2", root=tmp_path)[0] == [(8.0, 9.0)]
    assert ev.gt_segments(gt, "S3", root=tmp_path)[0] == []

    # 任意の音声を無音で区切ったもの（eval/segments_<name>.json）＋ eval/_annot_audio/ の音声
    (tmp_path / "eval" / "segments_tail.json").write_text(
        json.dumps([{"id": "0", "start": 1.0, "end": 3.5}, {"id": "1", "start": 4.0, "end": 6.0}]),
        encoding="utf-8")
    (tmp_path / "eval" / "_annot_audio").mkdir()
    ev.save_wav_16k(tmp_path / "eval" / "_annot_audio" / "tail.wav", np.zeros(SR * 8, dtype="float32"))
    gt2 = tmp_path / "eval" / "gt_tail.json"
    gt2.write_text(json.dumps({"session": "tail", "labels": {"0": "S1", "1": "S1"}}), encoding="utf-8")
    segs, wav = ev.gt_segments(gt2, "S1", root=tmp_path)
    assert segs == [(1.0, 3.5), (4.0, 6.0)] and wav.name == "tail.wav"

    assert ev.parse_from_gt("黒田=eval/gt_tail.json/S1:data/pairs/tail.wav") == (
        "黒田", "eval/gt_tail.json", "S1", "data/pairs/tail.wav")
    assert ev.parse_from_gt("としや=eval/gt_s.json/S2") == ("としや", "eval/gt_s.json", "S2", None)
