#!/usr/bin/env python3
"""実会話1ランを診断する（録音後に最初に流すスクリプト）.

実会話はそう何度も録れないので、1本から取れる判断材料を一度に出す。
新形式の記録（判定の入力つき, handoff §23）が前提。

出す数字は4つ:

  A. 再生の忠実性   記録どおりの結論を本番コードが再現するか。100%でなければ
                    記録し足りない入力がある＝ID空間の統合（§24.3）に進めない
  B. クラスタ分裂   マイク直で pyannote のクラスタがどれだけ割れるか。
                    **項目2（土台が2つとも壊れている問題）に決着をつける数字**。
                    分裂が支配的なら、門番を増やす方向ではなく入力側
                    （マイク配置・複数マイク・DOA）に手を入れるべき
  C. 席の圧迫       想定話者数に対して席が足りているか、落ちているのはどの経路か。
                    sysレコードのバグ修正（`51884b0`）が効いているかもここで見る
  D. 声紋層の健全性 ラベル不純・鋳造・鋳造リンクの発火状況

**Phase 0 から何が変わったか**: 窓は `decide_speaker` の先頭で記録されるので、
**声紋が勝った発話でもクラスタの所属が残る**。Phase 0 ではクラスタ層の観測が
「diarization が勝った発話」に限られ、蓄積も分裂も過小にしか測れなかった
（§7.4 の「案Aに構造的に不利」の原因）。その盲点が無くなっている。

使い方:
    uv run python eval/diagnose_live_session.py --session 2026-08-01_1030
    uv run python eval/diagnose_live_session.py --session X --ab    # 鋳造リンクA/B
    uv run python eval/diagnose_live_session.py --session X --gt eval/gt_X.json
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402

from das.asr.live._constants import _BACKCHANNEL_RE, UNSURE_SPEAKER  # noqa: E402


def read_diag(session: str, root: Path | None = None) -> dict:
    """diag を種類ごとに仕分けて返す."""
    root = root or ROOT / "transcripts"
    out: dict = {"config": {}, "utts": [], "cluster_naming": [],
                 "constrain_drop": [], "mint_link": [], "echo_drop": []}
    for line in _gtlib.read_jsonl(root / f"{session}.diag.jsonl"):
        t = line.get("type")
        if t == "session_config":
            out["config"] = line
        elif t == "cluster_naming":
            out["cluster_naming"].append(line)
        elif t == "constrain_drop":
            out["constrain_drop"].append(line)
        elif t == "mint_cluster_link":
            out["mint_link"].append(line)
        elif t == "echo_drop":
            out["echo_drop"].append(line)
        elif t is None and "label" in line and "key" in line:
            out["utts"].append(line)
    texts = {}
    for r in _gtlib.read_jsonl(root / f"{session}.turns.jsonl"):
        texts.setdefault(r["ms"], r.get("text", ""))
    out["texts"] = texts
    return out


def _is_bc(text: str) -> bool:
    return bool(_BACKCHANNEL_RE.match(text.strip()))


def report_fidelity(session: str, rec: dict) -> None:
    print("\n== A. 再生の忠実性 ==")
    utts = rec["utts"]
    with_window = sum(1 for u in utts if "diar" in u)
    with_flags = sum(1 for u in utts if "enr" in u)
    if not with_flags:
        print("  ✗ 判定の入力が記録されていない（2026-07-25 より前のコードで録れたラン）")
        print("    → このランでは再生できない。新しいコードで録り直しが必要")
        return
    print(f"  classify条件つき: {with_flags}/{len(utts)}発話")
    print(f"  diarization窓つき: {with_window}/{len(utts)}発話"
          f"（窓が無い＝その区間に話者区間が届いていない発話）")
    print(f"  構成の記録: {'あり' if rec['config'] else '✗ なし'}")
    print(f"  → 忠実性の確認: uv run python eval/replay_live_attribution.py "
          f"--session {session}")
    print("    「記録との自己一致」が100%なら、この記録で設計変更を検証できる")


def report_fragmentation(rec: dict) -> None:
    """クラスタ分裂の実測（項目2の決着をつける数字）."""
    print("\n== B. クラスタ分裂（マイク直の土台の壊れ方） ==")
    utts = rec["utts"]
    if not any("diar" in u for u in utts):
        print("  ✗ 窓が記録されていないため測定不可")
        return
    # 窓は decide_speaker の先頭で記録されるので、声紋が勝った発話でも
    # クラスタの所属が残る（Phase 0 の盲点が無い）
    sec_by_cluster: dict[str, float] = defaultdict(float)
    utt_by_cluster: Counter = Counter()
    for u in utts:
        dur = max(0, int(u["end"]) - int(u["ms"])) / 1000.0
        for src, spk, _s, _e in u.get("diar", []):
            key = f"{src}:{spk}"
            sec_by_cluster[key] += dur
            utt_by_cluster[key] += 1
    if not sec_by_cluster:
        print("  クラスタが1つも観測されていない（diarization が供給されていない）")
        return
    total = sum(sec_by_cluster.values())
    ordered = sorted(sec_by_cluster.items(), key=lambda kv: -kv[1])
    # 発話時間の90%を覆うのに何クラスタ要るか＝「実質のクラスタ数」
    acc, effective = 0.0, 0
    for _k, v in ordered:
        acc += v
        effective += 1
        if acc >= total * 0.9:
            break
    expected = rec["config"].get("diarization_max_speakers")
    print(f"  観測クラスタ数: {len(ordered)}")
    print(f"  実質クラスタ数（発話時間の90%を覆う数）: {effective}")
    if expected:
        print(f"  想定話者数: {expected} → 分裂率: {effective / expected:.1f}倍"
              f"（1.0が理想。Chibaのスピーカー再生は約1.0〜1.5、"
              f"実会話1723は約5.0）")
    print("  クラスタ別の内訳（上位8）:")
    for k, v in ordered[:8]:
        print(f"    {k:<24} {v:6.1f}秒 / {utt_by_cluster[k]:3d}発話"
              f" ({v / total:5.1%})")
    if len(ordered) > 8:
        rest = sum(v for _k, v in ordered[8:])
        print(f"    （残り{len(ordered) - 8}クラスタ 計{rest:.1f}秒 "
              f"{rest / total:.1%}）")
    print("  読み方: 実質クラスタ数が想定話者数に近ければ土台は健全で、"
          "帰属の改善は声紋層・門番の側で効く。")
    print("          大きく上回るなら分裂が支配的で、門番を増やしても頭打ち"
          "——入力側（マイク配置・複数マイク・DOA）の問題。")


def report_seats(rec: dict) -> None:
    print("\n== C. 席の圧迫 ==")
    utts = rec["utts"]
    texts = rec["texts"]
    non_bc = [u for u in utts if not _is_bc(texts.get(u["ms"], ""))]
    finals = Counter(str(u.get("final_key")) for u in non_bc)
    unsure = finals.get(UNSURE_SPEAKER, 0)
    n = len(non_bc) or 1
    seats = {k for k in finals if k != UNSURE_SPEAKER}
    expected = rec["config"].get("diarization_max_speakers")
    print(f"  実質発話（相槌除き）: {n}")
    print(f"  未確定: {unsure / n:.1%}")
    print(f"  席を持ったキー: {len(seats)}" + (f" / 想定 {expected}" if expected else ""))
    for k, v in finals.most_common(8):
        print(f"    {k:<12} {v:4d}発話 ({v / n:5.1%})")
    drops = rec["constrain_drop"]
    if drops:
        by_key = Counter(d["key"] for d in drops)
        print(f"  上限で落ちたキー: {len(by_key)}種 / 延べ{len(drops)}回")
        for k, v in by_key.most_common(5):
            print(f"    {k:<12} {v}回")
        print("    → 実在者が締め出されているなら想定話者数を上げる。"
              "分裂クラスタが落ちているだけなら想定どおり")
    else:
        print("  上限で落ちたキー: なし（席は足りていた）")


def report_voiceprint(rec: dict) -> None:
    print("\n== D. 声紋層の健全性 ==")
    utts = rec["utts"]
    kinds = Counter(u.get("kind") for u in utts)
    order = ["声紋一致", "補正", "自動登録", "合流", "蓄積中", "ラベル継続",
             "ラベル不純", "継続不可", "照合なし", "重なりスキップ",
             "純度保留", "話者数上限", "声紋計算不可", "AI声紋一致"]
    n = len(utts) or 1
    for k in order:
        if kinds.get(k):
            print(f"    {k:<10} {kinds[k]:4d} ({kinds[k] / n:5.1%})")
    other = {k: v for k, v in kinds.items() if k not in order}
    for k, v in other.items():
        print(f"    {k!s:<10} {v:4d} ({v / n:5.1%})")
    impure = kinds.get("ラベル不純", 0)
    print(f"  ラベル不純率: {impure / n:.1%}"
          "（Chiba最悪の1032は約50%。高いほどSonioxが話者を混載している）")
    confirms = [e for e in rec["cluster_naming"]
                if isinstance(e.get("match"), list)]
    declined = [e for e in rec["cluster_naming"]
                if e.get("kind") == "確定見送り(低確信)"]
    print(f"  クラスタ確定の試行: 成立{len(confirms)} / 見送り{len(declined)}")
    for e in declined[:5]:
        print(f"    見送り {e.get('cluster')} → {e.get('name')}"
              f" sim={e.get('sim')} (要 {e.get('need')})")
    links = rec["mint_link"]
    if rec["config"].get("vp_mint_cluster_link"):
        print(f"  鋳造リンク: {len(links)}件成立")
        for e in links:
            print(f"    {e.get('seat')} → {e.get('name')} 対称類似{e.get('sim')}")
        print("    → 見送りが多いのにリンクが0件なら、閾値0.50が実会話で"
              "厳しすぎる可能性（Phase 0 は記録16本で校正）")
    else:
        print("  鋳造リンク: 無効で録れたラン（--vp-mint-cluster-link 未指定）")


def report_ab(session: str, rec: dict, gt: Path | None) -> None:
    """同じ記録を鋳造リンク on/off で流し、差を出す."""
    import tempfile

    import replay_live_attribution as rla
    print("\n== E. 鋳造リンクのA/B（同じ記録・構成だけ変える） ==")
    full = rla.load_session(session)
    if not rla.has_replayable_inputs(full["utts"]):
        print("  ✗ 再生できない記録のため実施不可")
        return
    tmp = Path(tempfile.mkdtemp(prefix="diag_ab_"))
    rows_by = {}
    for label, ov in (("無効", {"vp_mint_cluster_link": False}),
                      ("有効", {"vp_mint_cluster_link": True})):
        rows = rla.replay(full, ov, tmp / label)
        rows_by[label] = rows
        non_bc = [r for r in rows if not r["bc"]]
        n = len(non_bc) or 1
        uns = sum(1 for r in non_bc if r["pred"] == UNSURE_SPEAKER)
        seats = {r["pred"] for r in non_bc if r["pred"] != UNSURE_SPEAKER}
        line = f"  {label}: 未確定 {uns / n:5.1%} / 席 {len(seats)}"
        sc = rla.score(rows, gt, session)
        if sc:
            line += f" / 実質 {sc['acc']:.1%} / 誤帰属 {sc['wrong']:.1%}"
        print(line)
    a, b = rows_by["無効"], rows_by["有効"]
    changed = sum(1 for x, y in zip(a, b, strict=True) if x["pred"] != y["pred"])
    print(f"  結論の変わった発話: {changed}/{len(a)}")
    print("  読み方: Phase 0 の予測は 実質+9pt / 未確定-15pt（GT11本平均）。"
          "実会話で同じ向きなら既定化してよい")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--session", required=True)
    p.add_argument("--gt", default=None, help="GT JSON（あれば A/B に精度も出す）")
    p.add_argument("--ab", action="store_true",
                   help="鋳造リンクのA/Bも実施（声紋モデルを読むため数分かかる）")
    args = p.parse_args(argv)

    rec = read_diag(args.session)
    print(f"# セッション {args.session}: {len(rec['utts'])}発話")
    print(f"# 構成: {rec['config'] or '（記録なし）'}")
    report_fidelity(args.session, rec)
    report_fragmentation(rec)
    report_seats(rec)
    report_voiceprint(rec)
    if args.ab:
        report_ab(args.session, rec, Path(args.gt) if args.gt else None)
    else:
        print("\n（--ab を付けると鋳造リンクのA/Bも実施します）")


if __name__ == "__main__":
    main()
