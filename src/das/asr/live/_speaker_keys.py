"""話者キーの種別を判定する語彙（唯一の出所）.

このシステムには話者を指す名前空間が5つある:

  ``#ラベル``   STTラベル由来のプレースホルダ（声紋の裏付けがまだ無い）
  ``@diar:N``   外部diarizationの生クラスタに発行した匿名キー
  ``人物N``     声紋が鋳造した匿名の戸籍（セッション限り）
  実名          ユーザーが付けた名前（voices.json に永続化される）
  参加者A       表示ラベル（``SessionState.anonymous_labels``）

従来この5空間の判別は ``key.startswith("#")`` のような文字列検査として
6ファイル・約30箇所に散らばっていた。同じ述語が3通り書かれている箇所もあり
（``SessionState._is_anonymous_speaker_key`` / ``_speaker_policy`` の
``is_reliable_human_speaker`` / 各所の直書き）、片方だけ直る事故の温床だった。
本モジュールはその語彙を1箇所に集める。

**なぜ「1つのIdentity型」にしないのか**: 5空間の統合は帰属の中核を6モジュール
横断で書き換える変更で、失敗すると「話者の帰属が微妙にずれる」——単体テストが
最も捕まえにくい壊れ方をする。安全網（記録からの本番コード再生, §23）は
新しい記録が必要で、まだ実会話を録れていない。まず語彙を1箇所に寄せておき、
統合そのものは再生で検証できるようになってから行う（§24）。

命名の約束: ``is_*_key`` はキー（records の speaker に入る値）を判定する。
``looks_like_system_name`` だけは**ユーザーが付けた名前**を見る別の問いなので
名前を分けてある（下記 docstring 参照）。
"""
from __future__ import annotations

import re

from ._constants import AGENT_SPEAKER, UNSURE_SPEAKER

# STTラベル由来のプレースホルダ（声紋の裏付けが無い暫定キー）
LABEL_PREFIX = "#"
# 外部diarizationの生クラスタに発行する匿名キー
CLUSTER_PREFIX = "@diar:"
# 声紋が鋳造する匿名の戸籍。VoiceProfiles が採番する形（話者N は含めない——
# システムが鋳造するのは 人物N だけで、話者N はユーザーが打ちうる名前の側）
MINTED_RE = re.compile(r"^人物\d+$")
# ユーザーが付けた名前のうち、システムの仮名に見えるもの。人物N に加えて
# 話者N も含む（表示ラベルの解放判定で「実名を付けた」と誤認しないため）
SYSTEM_LOOKING_NAME_RE = re.compile(r"(話者|人物)\d+")
# AI（ファシリテーター／パートナー）の声紋キー。__NAME__ の形
AI_KEY_PREFIX = AI_KEY_SUFFIX = "__"

# 人間の参加者として席・発言量・声かけの対象にしない特別なキー
NON_PARTICIPANT_KEYS = frozenset({AGENT_SPEAKER, "パートナー", UNSURE_SPEAKER})


def is_label_key(key: object) -> bool:
    """STTラベル由来のプレースホルダ（``#1`` 等）か."""
    return str(key).startswith(LABEL_PREFIX)


def is_cluster_key(key: object) -> bool:
    """外部diarizationに発行した匿名キー（``@diar:1`` 等）か."""
    return str(key).startswith(CLUSTER_PREFIX)


def is_minted_key(key: object) -> bool:
    """声紋が鋳造した匿名の戸籍（``人物1`` 等）か.

    「セッション限りで、voices.json に永続化しない」対象の判定に使う。
    """
    return MINTED_RE.match(str(key)) is not None


def is_ai_key(key: object) -> bool:
    """AI の声紋キー（``__AI__`` / ``__PARTNER__``）か."""
    k = str(key)
    return k.startswith(AI_KEY_PREFIX) and k.endswith(AI_KEY_SUFFIX)


def is_provisional_key(key: object) -> bool:
    """まだ誰とも結び付いていない暫定キー（``#ラベル`` か ``@diar:N``）か.

    表示ラベルの割り当て・参加人数の勘定・「AIが個人名で扱ってよいか」の判定は
    すべてこの述語を見る（従来3箇所で別々に書かれていた）。
    """
    return is_label_key(key) or is_cluster_key(key)


def is_person_key(key: object) -> bool:
    """声紋層が「実在の人物」として同定したキー（``人物N`` か 実名）か.

    暫定キー（``#ラベル`` / ``@diar:N``）・AIキー・非参加者キーの**どれでもない**
    キーがこれに当たる。席上限で落ちた発話を席の実音声で寄せ直してよいか
    （seat_audio §27）の判定に使う——寄せ直しの前提「落ちたキーは席持ちの
    分裂」は暫定キーにしか成り立たず、人物キーが落ちている場合は声紋層が
    「席に居ない実在の人物」という証拠を出している（rehacq検証 2026-08-01:
    冒頭の雑音声が席を先取りすると、3人目の実在話者 2,587字が丸ごと
    既存席へ押し込まれた。handoff §49）。
    """
    k = str(key)
    return (k not in NON_PARTICIPANT_KEYS and not is_provisional_key(k)
            and not is_ai_key(k))


def looks_like_system_name(name: object | None) -> bool:
    """**ユーザーが付けた名前**がシステムの仮名に見えるか（``人物3`` ``話者2`` 等）.

    ``is_minted_key`` とは別の問い。あちらは「システムが鋳造したキーか」、
    こちらは「ユーザーが実名を付けたと見なしてよいか」で、後者には 話者N も
    含める（ユーザーが打ちうるため）。実名を付けたときだけ表示ラベルの文字を
    解放する規則（handoff の「実名は文字を解放」）が、仮名の入力で誤発動する
    のを防ぐ。
    """
    if not name:
        return False
    return SYSTEM_LOOKING_NAME_RE.fullmatch(str(name)) is not None


def is_anonymous(key: object) -> bool:
    """表示上まだ匿名の扱いをする対象か（暫定キー or システムの仮名）."""
    return is_provisional_key(key) or looks_like_system_name(key)
