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


def test_ai_voice_keys_are_never_persisted(tmp_path):
    """AI声紋（__AI__/__PARTNER__）は voices.json に書かない.

    エージェントが毎セッション tracker.profiles に書き込む一時キーで、
    永続化すると次回起動時に「保存済みプロファイル」として UI に並ぶ
    （レビュー 2026-07-30）。
    """
    path = tmp_path / "voices.json"
    vp = VoiceProfiles(path=str(path), model="redimnet",
                       embedder=lambda w: np.array([0.0, 1.0]))
    vp.profiles["__AI__"] = np.array([1.0, 0.0])
    vp.profiles["人物1"] = np.array([0.0, 1.0])
    vp.profiles["田中"] = np.array([0.5, 0.5])
    vp._persist()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "田中" in saved
    assert "__AI__" not in saved
    assert "人物1" not in saved
