"""scripts/enroll_voices.py と --activate（事前登録の再生評価の経路）のテスト."""
from __future__ import annotations

import importlib.util
import json
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
