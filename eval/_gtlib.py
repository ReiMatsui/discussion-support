#!/usr/bin/env python3
"""eval スクリプト共通のGT処理（JSONL読み込み・タイムライン突合・最適1:1対応）.

eval_speaker_gt.py / replay_attribution.py / transplant_gt.py に重複していた
採点系の中核を一本化する（2026-07-17 リファクタ。それまで3実装が
「支配80%・カバレッジ30%・最適1:1対応の総当たり」を別々に持ち、片方だけ
直る事故の温床だった）。prep_chiba.py はGTの生成側・run_chiba.py は
eval_speaker_gt へ委譲するオーケストレータで、採点ロジックを持たないため
本モジュールの対象外。

採点数値の互換維持のための設計メモ:
  - 「未確定」の番兵はスクリプトごとに値が違う（eval_speaker_gt の
    "未確定" は turns.jsonl の表示ラベル、replay の "?"=UNSURE_SPEAKER は
    classify の生キー）ため、best_mapping は番兵を引数で受け取る
  - best_mapping の探索順（real は sorted、gts は呼び出し側の並び）と
    tie-break（厳密な > 比較＝先勝ち）は旧実装と同一。accuracy 値は
    並びに依存しないが、同率時に報告される対応表は並びに依存するため
  - しきい値定数（0.3 / 0.8）は3実装で厳密一致していたものをそのまま採用
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 本体（das）の相槌判定などを uv 外の直接実行でも読めるようにする
sys.path.insert(0, str(ROOT / "src"))


def read_jsonl(path) -> list[dict]:
    """1行1JSONのファイル（turns.jsonl / diag.jsonl）を読む."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_backchannel_re():
    """本体と同じ相槌判定の正規表現を返す（das が無い環境ではフォールバック）."""
    try:
        from das.asr.live._constants import _BACKCHANNEL_RE
        return _BACKCHANNEL_RE
    except ImportError:   # 最後の砦: スタンドアロン用フォールバック（本体と同期すること）
        import re
        return re.compile(
            r'^[\s、。,.!?！？]*'
            r'(うん|ふん|ふーん|へー|ほー|おー|あー|えー'
            r'|はい|ええ|そう|そっか|そうだね|そうですね|そうですか'
            r'|なるほど|確かに|分かる|わかる|分かります|わかりました'
            r'|了解|オッケー|OK)'
            r'[\s、。,.!?！？うんはいええそっかなるほど確かに]*$',
            re.IGNORECASE,
        )


def gt_timeline(gt_turns, labels) -> dict[str, list[tuple[int, int]]]:
    """区間単位のGT（labels形式）を話者→時間区間リストのタイムラインに変換する.

    区間単位のGTは特定の区切りに紐づくため、別の区切りのランを採点するには
    「誰がいつ喋っていたか」の時間表現に直すのが正しい（区切りが違う発話に
    単一の正解を押し付けない）。
    """
    tl: dict[str, list[tuple[int, int]]] = {}
    for g in gt_turns:
        c = labels.get(str(g["turn_id"])) or labels.get(g["turn_id"])
        if c in ("S1", "S2", "S3"):
            tl.setdefault(c, []).append((g["ms"], g["end_ms"]))
    return tl


def gt_code_by_timeline(ms: int, end_ms: int,
                        tl: dict[str, list[tuple[int, int]]]) -> str | None:
    """発話区間 [ms, end_ms] で支配的（80%以上）なGT話者を返す.

    GTラベル済み時間が発話長の3割未満なら None（正解範囲外＝採点対象外）、
    支配的話者がいなければ "MULTI"（複数人が混在する区間＝単一正解を
    割り当てられないため採点対象外として件数報告のみ）。
    """
    dur = max(1, end_ms - ms)
    ovs = {c: sum(max(0, min(end_ms, b) - max(ms, a)) for a, b in ivs)
           for c, ivs in tl.items()}
    total = sum(ovs.values())
    if total < dur * 0.3:
        return None
    c, top = max(ovs.items(), key=lambda x: x[1])
    return c if top >= total * 0.8 else "MULTI"


def best_mapping(pairs: list[tuple[str, str]], gts, *,
                 unsure: str) -> tuple[float, dict]:
    """最適1:1対応（未確定は常に不正解扱い）での帰属精度と対応表を返す.

    pairs は (システムラベル, GT話者) の列。unsure（未確定の番兵）は対応の
    候補から除外され、どの対応でも不正解として分母にだけ入る。探索は
    permutation の総当たり（ラベル数・話者数とも実用上3前後のため十分速い）。
    """
    real = sorted({p for p, _ in pairs if p != unsure})
    best_acc, best_map = 0.0, {}
    for k in range(min(len(gts), len(real)) + 1):
        for perm in permutations(real, k):
            for gsel in permutations(gts, k):
                m = dict(zip(perm, gsel, strict=False))
                acc = sum(1 for p, g in pairs if m.get(p) == g) / len(pairs)
                if acc > best_acc:
                    best_acc, best_map = acc, m
    return best_acc, best_map
