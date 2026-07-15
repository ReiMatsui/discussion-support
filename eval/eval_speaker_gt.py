#!/usr/bin/env python3
"""話者帰属の正解(GT)評価スクリプト.

使い方:
    uv run python eval/eval_speaker_gt.py eval/gt_2026-07-14_142016.json
    uv run python eval/eval_speaker_gt.py eval/gt_2026-07-14_142016.json 2026-07-14_180000

GT は eval/gt_annotator_*.html で作成した JSON（labels: {turn_id: S1|S2|S3|MULTI|UNK}）。
対応する transcripts/<session>.turns.jsonl / .diag.jsonl を自動で読む。

第2引数に別セッション名を渡すと、同じ音声を --wav で再実行したランを評価できる:
GTの発話区間（元セッションの ms〜end_ms）と新ランの発話を時間重なりで突合するため、
発話の区切りが多少揺れても再アノテーション不要（同一音声タイムラインが前提）。

出力: 混同行列、システムラベルの純度、GT話者ごとの分裂状況、未確定率、
最適1:1対応での帰属精度、名寄せイベントのタイムライン。
設計: docs/design/handoff_2026-07-14_unregistered_speakers.md §4 の定量化。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from das.asr.live._constants import _BACKCHANNEL_RE  # noqa: E402  本体と同一の相槌判定

UNSURE = "未確定"


def load(gt_path: str, session_override: str | None = None):
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent.parent / "transcripts"

    def read_turns(session):
        return [json.loads(l)
                for l in open(root / f"{session}.turns.jsonl", encoding="utf-8")]

    if gt.get("kind") == "timeline":
        # タイムライン形式GT（eval/fetch_callhome_jpn.py 等が生成）:
        # 発話区間ラベルでなく「誰がいつ喋ったか」を直接持つ。採点対象の
        # セッション名は第2引数で必ず指定する。
        if not session_override:
            sys.exit("タイムライン形式GTでは採点対象セッション名を指定してください")
        session = session_override
        turns = read_turns(session)
        diag_path = root / f"{session}.diag.jsonl"
        diag = ([json.loads(l) for l in open(diag_path, encoding="utf-8")]
                if diag_path.exists() else [])
        return gt, None, turns, diag, session

    gt_turns = read_turns(gt["session"])  # GTのturn_id→時間区間の定義元
    session = session_override or gt["session"]
    turns = gt_turns if session == gt["session"] else read_turns(session)
    diag_path = root / f"{session}.diag.jsonl"
    diag = ([json.loads(l) for l in open(diag_path, encoding="utf-8")]
            if diag_path.exists() else [])
    return gt, gt_turns, turns, diag, session


def gt_timeline(gt_turns, labels):
    """GTを話者→時間区間リストのタイムラインに変換する.

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


def gt_code_by_timeline(t, tl):
    """発話 t の時間帯で支配的（80%以上）なGT話者を返す.

    ラベル済み時間が発話長の3割未満なら None（正解範囲外）、
    支配的話者がいなければ "MULTI"（複数人が混在する区間＝単一正解を
    割り当てられないため採点対象外として件数報告のみ）。
    """
    s, e = t["ms"], t["end_ms"]
    dur = max(1, e - s)
    ovs = {c: sum(max(0, min(e, b) - max(s, a)) for a, b in ivs)
           for c, ivs in tl.items()}
    total = sum(ovs.values())
    if total < dur * 0.3:
        return None
    c, top = max(ovs.items(), key=lambda x: x[1])
    return c if top >= total * 0.8 else "MULTI"


def main(gt_path: str, session_override: str | None = None) -> None:
    gt, gt_turns, turns, diag, session = load(gt_path, session_override)
    names = gt.get("speaker_names", {})
    is_timeline_gt = gt.get("kind") == "timeline"
    same_session = (not is_timeline_gt) and session == gt["session"]

    if is_timeline_gt:
        gt_codes = list(gt["speakers"])
        tl = {}
        for seg in gt["timeline"]:
            tl.setdefault(seg["speaker"], []).append(
                (seg["start_ms"], seg["end_ms"]))
    else:
        gt_codes = ["S1", "S2", "S3"]
        labels: dict[str, str] = gt["labels"]
        tl = gt_timeline(gt_turns, labels) if not same_session else None

    rows = []  # (turn_id, sys_label, gt_code, is_backchannel)
    n_uncovered = 0
    for t in turns:
        if same_session:
            code = labels.get(str(t["turn_id"])) or labels.get(t["turn_id"])
        else:
            code = gt_code_by_timeline(t, tl)
            if code is None:
                n_uncovered += 1
                continue
        if code:
            bc = bool(_BACKCHANNEL_RE.match(t.get("text", "").strip()))
            rows.append((t["turn_id"], t["speaker"], code, bc))
    if not rows:
        sys.exit("GTラベルとturnsが突合できません")

    single = [(i, s, g, b) for i, s, g, b in rows if g in gt_codes]
    n_multi = sum(1 for _, _, g, _ in rows if g == "MULTI")
    n_unk = sum(1 for _, _, g, _ in rows if g == "UNK")

    src = ("" if same_session else
           (f"（GT元: {gt['session']} タイムライン突合、"
            f"正解範囲外 {n_uncovered} 件除外）"))
    print(f"= セッション {session}{src}: GT付き {len(rows)}/{len(turns)} 発話"
          f"（単独話者 {len(single)}、複数人混在 {n_multi}、不明 {n_unk}）\n")

    # --- 混同行列（単独話者のみ） ---
    sys_labels = sorted({s for _, s, _, _ in single})
    gts = gt_codes
    conf = {s: Counter() for s in sys_labels}
    for _, s, g, _ in single:
        conf[s][g] += 1
    w = max(len(s) for s in sys_labels) + 2
    print("== 混同行列（行=システム, 列=正解） ==")
    print(" " * w + "  ".join(f"{names.get(g, g):>6}" for g in gts) + "   純度")
    for s in sys_labels:
        total = sum(conf[s].values())
        purity = max(conf[s].values()) / total if total else 0
        print(f"{s:<{w}}" + "  ".join(f"{conf[s][g]:>6}" for g in gts)
              + f"   {purity:.0%}" + ("  ←未確定" if s == UNSURE else ""))

    # --- GT話者ごとの分裂 ---
    print("\n== 正解話者ごとの行き先（分裂状況） ==")
    by_gt = defaultdict(Counter)
    for _, s, g, _ in single:
        by_gt[g][s] += 1
    for g in gts:
        c = by_gt[g]
        if not c:
            continue
        total = sum(c.values())
        parts = ", ".join(f"{s}:{n}" for s, n in c.most_common())
        main_label, main_n = c.most_common(1)[0]
        print(f"{names.get(g, g)}（{total}発話）→ {parts}")
        print(f"    最多ラベル {main_label} への集中度: {main_n/total:.0%}"
              f" ／ 未確定率: {c[UNSURE]/total:.0%}")

    # --- 最適1:1対応での帰属精度（未確定は常に不正解扱い） ---
    real = [s for s in sys_labels if s != UNSURE]
    best_acc, best_map = 0.0, {}
    for k in range(min(len(gts), len(real)) + 1):
        for perm in permutations(real, k):
            for gsel in permutations(gts, k):
                m = dict(zip(perm, gsel))
                acc = sum(1 for _, s, g, _ in single if m.get(s) == g) / len(single)
                if acc > best_acc:
                    best_acc, best_map = acc, m
    print(f"\n== 最適1:1対応での帰属精度: {best_acc:.0%} ==")
    for s, g in best_map.items():
        print(f"  {s} = {names.get(g, g)}")
    # 相槌内訳（ヘッドラインは従来どおり全発話）。相槌判定は本体と同じ
    # _BACKCHANNEL_RE。現行ルール（相槌は未確定に落とす）の妥当性を数字で
    # 見るための参考値で、対応表はヘッドラインの最適対応をそのまま使う。
    non_bc = [(i, s, g) for i, s, g, b in single if not b]
    if non_bc:
        acc_nb = sum(1 for _, s, g in non_bc if best_map.get(s) == g) / len(non_bc)
        n_bc = len(single) - len(non_bc)
        bc_ok = sum(1 for _, s, g, b in single if b and best_map.get(s) == g)
        bc_uns = sum(1 for _, s, _, b in single if b and s == UNSURE)
        print(f"  相槌内訳: 全発話 {best_acc:.1%} (n={len(single)})"
              f" ／ 相槌除き {acc_nb:.1%} (n={len(non_bc)})"
              f" ／ 相槌 {n_bc}件（未確定 {bc_uns}、正解 {bc_ok}）")
    unsure_rate = sum(1 for _, s, _, _ in single if s == UNSURE) / len(single)
    n_wrong = sum(1 for _, s, g, _ in single
                  if s != UNSURE and s in best_map and best_map[s] != g)
    n_unmapped = sum(1 for _, s, _, _ in single
                     if s != UNSURE and s not in best_map)
    print(f"  内訳: 未確定 {unsure_rate:.0%} ／ 誤帰属 {n_wrong/len(single):.0%}"
          f" ／ 対応外ラベル {n_unmapped/len(single):.0%}")

    # --- diag: 名寄せ・声紋イベント ---
    if diag:
        print("\n== diag イベント ==")
        kinds = Counter(d.get("type") or d.get("kind") for d in diag)
        print("  件数:", dict(kinds))
        for d in diag:
            if d.get("type") == "cluster_naming":
                m = d.get("match")
                sim = f" match={m[0]} sim={m[1]:.3f}" if m else " match=なし"
                extra = {k: v for k, v in d.items()
                         if k not in ("type", "ms", "end", "match")}
                print(f"  [{d['ms']/1000:7.1f}s] cluster_naming{sim} {extra}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
