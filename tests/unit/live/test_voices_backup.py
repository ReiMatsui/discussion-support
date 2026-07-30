"""voices.json の永続化と、リネーム/リセット時の台帳一貫性のテスト."""
import json

import numpy as np

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


def _vp(tmp_path):
    return VoiceProfiles(path=str(tmp_path / "v.json"), model="redimnet",
                         embedder=lambda w: np.array([0.0, 1.0]))


def test_rename_updates_the_label_purity_history(tmp_path):
    """リネーム後もラベル健全性の履歴が1人物のままであること.

    履歴に旧キーが残ると、次の照合成功（新キー）で「直近windowに2人物」と
    誤判定され、正しい命名操作がラベル不純→未確定を誘発する
    （レビュー 2026-07-30）。
    """
    vp = _vp(tmp_path)
    vp.label_hist["1"] = ["人物1", "人物1", "人物1"]
    vp.profiles["人物1"] = np.array([0.0, 1.0])
    vp._active_keys.add("人物1")
    assert vp.enroll("人物1", "田中") is not None
    assert vp.label_hist["1"] == ["田中", "田中", "田中"]
    assert vp._label_pure("1")

    vp.label_hist["2"] = ["人物2", "人物2"]
    vp.profiles["人物2"] = np.array([1.0, 0.0])
    vp.profiles["佐藤"] = np.array([1.0, 0.0])
    assert vp.remap("人物2", "佐藤") is True
    assert vp.label_hist["2"] == ["佐藤", "佐藤"]


def test_meeting_reset_drops_minted_profiles_entirely(tmp_path):
    """会議リセットで 人物N のプロファイル実体も消えること.

    実体が残ると n_anon=0 に戻した後、次の会議でまだ鋳造されていない 人物N を
    命名したとき、前の会議の不在者の声紋が実名登録・永続化される
    （レビュー 2026-07-30）。
    """
    vp = _vp(tmp_path)
    vp.profiles["人物1"] = np.array([0.0, 1.0])
    vp.profiles["田中"] = np.array([1.0, 0.0])
    vp.profiles["__AI__"] = np.array([0.5, 0.5])
    vp._active_keys |= {"人物1", "田中", "__AI__"}
    vp.reset_session()
    assert "人物1" not in vp.profiles, "前会議の匿名戸籍が残っている"
    assert "田中" in vp.profiles          # 実名は次の会議へ引き継ぐ
    assert "__AI__" in vp.profiles        # AI声紋はエコー除去に使うため維持
