import json, sys, numpy as np
sys.path.insert(0, "src")
from das.asr.live._voice_profiles import VoiceProfiles

def test_foreign_model_profiles_are_backed_up_not_wiped(tmp_path):
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({"_model": "resemblyzer", "田中": [1.0, 0.0]}),
                    encoding="utf-8")
    vp = VoiceProfiles(path=str(path), model="redimnet",
                       embedder=lambda w: np.array([0.0, 1.0]))
    assert vp.profiles == {}                      # 互換性が無いので読み込まない
    vp._persist()                                 # enroll等の書き込み相当
    bak = tmp_path / "voices.json.resemblyzer.bak"
    assert bak.exists(), "旧モデルの台帳が退避されていない"
    assert "田中" in json.loads(bak.read_text(encoding="utf-8"))
    assert "田中" not in json.loads(path.read_text(encoding="utf-8"))
    vp._persist()                                 # 2回目は退避を上書きしない
    assert "田中" in json.loads(bak.read_text(encoding="utf-8"))
