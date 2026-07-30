"""話者キーの語彙（_speaker_keys）のテスト.

このシステムには話者を指す名前空間が5つあり（#ラベル / @diar:N / 人物N /
実名 / 表示ラベル）、その判別が6ファイル・約30箇所に文字列検査として
散らばっていた。同じ述語が3通り書かれている箇所もあった（handoff §24）。
ここでは語彙の意味を固定し、**特に紛らわしい2つの問い**を取り違えないよう
境界を明示する:

  is_minted_key        「システムが鋳造したキーか」（人物N のみ）
  looks_like_system_name「ユーザーが付けた名前が仮名に見えるか」（人物N と 話者N）

前者は「voices.json に永続化しない・リセットで落とす」対象を決め、後者は
「実名を付けたので表示ラベルの文字を解放してよいか」を決める。別の問いなので
対象集合が違うのは意図的で、統合してはいけない。
"""
from __future__ import annotations

import pytest

from das.asr.live._speaker_keys import (
    is_ai_key,
    is_cluster_key,
    is_label_key,
    is_minted_key,
    is_provisional_key,
    looks_like_system_name,
)


@pytest.mark.parametrize(("key", "expected"), [
    ("#1", True), ("#SPEAKER_00", True),
    ("@diar:1", False), ("人物1", False), ("田中", False), ("?", False),
])
def test_is_label_key(key, expected):
    assert is_label_key(key) is expected


@pytest.mark.parametrize(("key", "expected"), [
    ("@diar:1", True), ("@diar:12", True),
    ("#1", False), ("人物1", False), ("田中", False),
])
def test_is_cluster_key(key, expected):
    assert is_cluster_key(key) is expected


@pytest.mark.parametrize(("key", "expected"), [
    ("人物1", True), ("人物12", True),
    ("話者1", False),      # 鋳造するのは 人物N だけ（話者N はユーザー入力側）
    ("人物", False), ("人物1さん", False), ("@diar:1", False), ("田中", False),
])
def test_is_minted_key(key, expected):
    assert is_minted_key(key) is expected


@pytest.mark.parametrize(("key", "expected"), [
    ("__AI__", True), ("__PARTNER__", True),
    ("__", True),          # 前後が同じ印なので形としては AI キー
    ("_AI_", False), ("人物1", False), ("#1", False),
])
def test_is_ai_key(key, expected):
    assert is_ai_key(key) is expected


@pytest.mark.parametrize(("key", "expected"), [
    ("#1", True), ("@diar:1", True),
    ("人物1", False),      # 鋳造済み＝声紋の裏付けがあるので暫定ではない
    ("田中", False), ("?", False),
])
def test_is_provisional_key(key, expected):
    """暫定キー＝まだ誰とも結び付いていないキー（席・声かけの判定の共通述語）."""
    assert is_provisional_key(key) is expected


@pytest.mark.parametrize(("name", "expected"), [
    ("人物3", True), ("話者2", True),
    ("田中", False), ("", False), (None, False),
    ("人物3さん", False),   # 完全一致のみ（部分一致で実名を仮名扱いしない）
])
def test_looks_like_system_name(name, expected):
    assert looks_like_system_name(name) is expected


def test_minted_key_and_system_looking_name_are_different_questions():
    """「鋳造したキーか」と「仮名に見える名前か」は別（統合してはいけない）.

    話者N はシステムが鋳造しないが、ユーザーが打ちうる仮名。前者に含めると
    ユーザーの付けた「話者1」が voices.json に永続化されなくなり、後者から
    外すと表示ラベルの文字が誤って解放される。
    """
    assert is_minted_key("話者1") is False
    assert looks_like_system_name("話者1") is True
    # 人物N は両方に該当する（鋳造もするし、仮名にも見える）
    assert is_minted_key("人物1") and looks_like_system_name("人物1")


def test_vocabulary_is_the_single_source_for_key_shapes():
    """キーの形の定義がこのモジュールに集約されている（写しを作らない）.

    VoiceProfiles.ANON は互換のために残しているが、実体は語彙側と同一物で
    なければならない（別々に書くと片方だけ直る事故になる）。
    """
    from das.asr.live._speaker_keys import MINTED_RE
    from das.asr.live._voice_profiles import VoiceProfiles
    assert VoiceProfiles.ANON is MINTED_RE
