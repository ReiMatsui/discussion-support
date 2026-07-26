#!/usr/bin/env python3
"""話者帰属の損失を「原因別」に分解する（どこを直せば何pt返るかの見積り）.

これまで帰属の成績は「実質○%・誤帰属○%・未確定○%」という**総量**でしか
見えておらず、「未確定の何割がラベル不純由来なのか、席不足なのか、
ヒステリシス未達なのか」が分からなかった。そのため改善の優先順位が
「たぶんこれが効きそう」の域を出なかった。

本スクリプトは、GT付きランの1発話ごとに「正解／誤帰属／未確定」を判定し、
未確定と誤帰属をさらに**発生した層**で分類する。上限（その原因を完全に
解消したら何pt返るか）が読めるので、投資先を数字で選べる。

分類（相槌を除く実質発話が分母。相槌は設計上つねに未確定＝§13.2 の決定）:

  正解
  誤帰属        最終キーが GT と一致しない。声紋層の kind 別に内訳を出す
  未確定/席落ち   上流は決めていたのに constrain（参加人数上限・closed roster）
                で落ちた。**席の問題**——上限設定か二重帳簿が原因
  未確定/ラベル不純 声紋層が「ラベル不純」で棄権。**Soniox の話者混載**が原因
  未確定/継続不可  対応先が AI 声紋や無効化済みで継続できず棄権
  未確定/蓄積待ち  声紋もクラスタも育っておらず決められない（ヒステリシス未達・
                蓄積中・照合なし）。**時間が解決する類**で、序盤に集中するはず
  未確定/その他

「上限」の読み方: たとえば「未確定/ラベル不純 20%」なら、ラベル不純を完全に
解消しても**最大で** +20pt しか返らない（しかも全部が正解になる前提なので
実際はもっと少ない）。逆にここが 3% なら、そこに投資しても意味がない。

使い方:
    uv run python eval/decompose_attribution.py --all
    uv run python eval/decompose_attribution.py --run 2026-07-20_1723 --detail
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402

from das.asr.live._constants import _BACKCHANNEL_RE, UNSURE_SPEAKER  # noqa: E402

GT_CODES = ("S1", "S2", "S3")

# 声紋層が棄権した理由（VoiceProfiles._classify の kind）→ 分解上のラベル
_ABSTAIN = {
    "ラベル不純": "未確定/ラベル不純",
    "継続不可": "未確定/継続不可",
    "未確定": "未確定/継続不可",
    "話者数上限": "未確定/席落ち",
    "重なりスキップ": "未確定/重なり",
}
# まだ育っていない（時間が解決しうる）
_IMMATURE = {"蓄積中", "照合なし", "ラベル継続", "声紋計算不可"}

ORDER = ["正解", "誤帰属", "未確定/席落ち", "未確定/ラベル不純", "未確定/重なり",
         "未確定/継続不可", "未確定/蓄積待ち", "未確定/その他"]


def load_run(run: str) -> tuple[list[dict], dict[int, str]] | None:
    """(diag発話, ms→GTコード) を返す（GT・diag が揃わないなら None）."""
    gt_path = ROOT / "eval" / f"gt_{run}.json"
    diag_path = ROOT / "transcripts" / f"{run}.diag.jsonl"
    turns_path = ROOT / "transcripts" / f"{run}.turns.jsonl"
    if not (gt_path.exists() and diag_path.exists() and turns_path.exists()):
        return None
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    labels = gt.get("labels") or {}
    code_by_ms: dict[int, str] = {}
    text_by_ms: dict[int, str] = {}
    for t in _gtlib.read_jsonl(turns_path):
        c = labels.get(str(t["turn_id"])) or labels.get(t["turn_id"])
        if c:
            code_by_ms[t["ms"]] = c
        text_by_ms.setdefault(t["ms"], t.get("text", ""))
    utts = [d for d in _gtlib.read_jsonl(diag_path)
            if d.get("type") is None and "label" in d and "key" in d]
    for u in utts:
        u["_text"] = text_by_ms.get(u["ms"], "")
    return utts, code_by_ms


def classify_outcome(u: dict, mapping: dict, code: str) -> str:
    """1発話の結末を分類する."""
    # final_key は 2026-07-14 以降の追加フィールド。無い旧ランでは constrain 前の
    # key で代用する（席落ちは分離できなくなるが、正解/誤帰属は測れる）
    final = str(u["final_key"] if u.get("final_key") is not None else u.get("key"))
    key = str(u.get("key"))
    kind = u.get("kind")
    if final != UNSURE_SPEAKER:
        return "正解" if mapping.get(final) == code else "誤帰属"
    # --- ここから未確定の原因分け ---
    if key != UNSURE_SPEAKER:
        # 上流は決めていたのに constrain で落ちた＝席の問題
        return "未確定/席落ち"
    if kind in _ABSTAIN:
        return _ABSTAIN[kind]
    if kind in _IMMATURE:
        return "未確定/蓄積待ち"
    return "未確定/その他"


def decompose(run: str) -> dict | None:
    loaded = load_run(run)
    if loaded is None:
        return None
    utts, code_by_ms = loaded
    rows = []
    for u in utts:
        code = code_by_ms.get(u["ms"])
        if code not in GT_CODES:
            continue
        if _BACKCHANNEL_RE.match(str(u["_text"]).strip()):
            continue      # 相槌は設計上つねに未確定（§13.2）。実質発話から除く
        rows.append((u, code))
    if not rows:
        return None
    def _final(u: dict) -> str:
        return str(u["final_key"] if u.get("final_key") is not None
                   else u.get("key"))
    _acc, mapping = _gtlib.best_mapping(
        [(_final(u), c) for u, c in rows], GT_CODES, unsure=UNSURE_SPEAKER)
    counts: Counter = Counter()
    wrong_by_kind: Counter = Counter()
    for u, code in rows:
        out = classify_outcome(u, mapping, code)
        counts[out] += 1
        if out == "誤帰属":
            wrong_by_kind[u.get("kind")] += 1
    n = len(rows)
    return {"run": run, "n": n,
            "has_final": any(u.get("final_key") is not None for u, _ in rows),
            "share": {k: counts.get(k, 0) / n for k in ORDER},
            "wrong_by_kind": wrong_by_kind}


# ------------------------------------------------------------------
# 反実仮想: 不純回収門番（§18.8）の対象 kind を広げたら何が起きるか
# ------------------------------------------------------------------

def endorsed(u: dict) -> bool:
    """その発話自身の声紋1位候補が回収先と一致し、弱い裏付けがあるか（§18.8）.

    本体の判定（_attribution.py ステップ3d）と同じ式:
        d["name"] == sp_id かつ sim >= CLUSTER_IMPURE_RECOVERY_ENDORSE_MIN_SIM
    diag の ``key`` が本体の ``sp_id``（constrain 前のクラスタ解決結果）に当たる。
    """
    from das.asr.live._constants import CLUSTER_IMPURE_RECOVERY_ENDORSE_MIN_SIM
    return (u.get("name") == u.get("key")
            and float(u.get("sim") or 0.0)
            >= CLUSTER_IMPURE_RECOVERY_ENDORSE_MIN_SIM)


def counterfactual(run: str, gate_kinds: set[str]) -> dict | None:
    """門番を ``gate_kinds`` にも適用したときの成績を出す.

    門番は**最終キーだけを差し替える**設計（台帳・蓄積の副作用は残す）なので、
    記録された最終キーを置き換えるだけでオフラインで意味論が一致する
    （_attribution.py の `_cluster_attribution` docstring 参照）。
    """
    loaded = load_run(run)
    if loaded is None:
        return None
    utts, code_by_ms = loaded
    rows = []
    for u in utts:
        code = code_by_ms.get(u["ms"])
        if code in GT_CODES and not _BACKCHANNEL_RE.match(str(u["_text"]).strip()):
            rows.append((u, code))
    if not rows:
        return None

    def _score(final_of) -> dict:
        pairs = [(final_of(u), c) for u, c in rows]
        _a, mapping = _gtlib.best_mapping(pairs, GT_CODES, unsure=UNSURE_SPEAKER)
        n = len(pairs)
        ok = sum(1 for f, c in pairs if mapping.get(f) == c)
        uns = sum(1 for f, _ in pairs if f == UNSURE_SPEAKER)
        return {"acc": ok / n, "unsure": uns / n,
                "wrong": (n - ok - uns) / n, "n": n}

    def _cur(u: dict) -> str:
        return str(u["final_key"] if u.get("final_key") is not None
                   else u.get("key"))

    def _cf(u: dict) -> str:
        cur = _cur(u)
        if (cur != UNSURE_SPEAKER and u.get("kind") in gate_kinds
                and not endorsed(u)):
            return UNSURE_SPEAKER
        return cur

    return {"run": run, "before": _score(_cur), "after": _score(_cf)}


def seat_recovery(run: str) -> dict | None:
    """席落ち（constrain で落ちた未確定）を回収したら何が起きるかを測る.

    「未確定を減らす」打ち手には2種類あり、**混ぜて考えると判断を誤る**:

      (a) 精度とのトレードオフ: 門番を緩める。未確定は減るが誤帰属が増える
      (b) トレードオフではない: 上流が既にキーを決めていたのに constrain
          （参加人数上限・closed roster）で落とした分を回収する。これは
          帳簿の問題であって判定の問題ではない

    本関数が測るのは (b) の上限。``final_key`` が未確定で ``key`` が
    決まっている発話について、``key`` をそのまま採用したらどうなるかを見る。
    上流のキーが GT と合っているなら、席の問題を直すだけで正解が増える
    （誤帰属は上流自身の誤り分しか増えない）。

    二重帳簿の根治（§21 の鋳造リンク）はまさにこの席を空ける施策であり、
    ここで返る値がその投資対効果の見積りになる。
    """
    loaded = load_run(run)
    if loaded is None:
        return None
    utts, code_by_ms = loaded
    rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
    rows = [(u, c) for u, c in rows if c in GT_CODES
            and not _BACKCHANNEL_RE.match(str(u["_text"]).strip())]
    if not rows or not any(u.get("final_key") is not None for u, _ in rows):
        return None      # final_key の無い旧ランでは席落ちを分離できない

    def _score(final_of) -> dict:
        pairs = [(final_of(u), c) for u, c in rows]
        _a, mapping = _gtlib.best_mapping(pairs, GT_CODES, unsure=UNSURE_SPEAKER)
        n = len(pairs)
        ok = sum(1 for f, c in pairs if mapping.get(f) == c)
        uns = sum(1 for f, _ in pairs if f == UNSURE_SPEAKER)
        return {"acc": ok / n, "unsure": uns / n,
                "wrong": (n - ok - uns) / n, "n": n}

    def _cur(u: dict) -> str:
        return str(u["final_key"])

    def _recovered(u: dict) -> str:
        # constrain 前のキーをそのまま採る＝席が足りていた世界
        return str(u["final_key"]) if str(u["final_key"]) != UNSURE_SPEAKER \
            else str(u.get("key"))

    dropped = sum(1 for u, _ in rows if str(u["final_key"]) == UNSURE_SPEAKER
                  and str(u.get("key")) != UNSURE_SPEAKER)
    return {"run": run, "before": _score(_cur), "after": _score(_recovered),
            "dropped": dropped}


def merge_ceiling(run: str) -> dict | None:
    """クラスタ分裂を「完全に」統合できたら何が起きるかの上限を測る.

    `seat_recovery` は席落ちを**そのまま**通す＝分裂したクラスタが別人として
    出るので、未確定が誤帰属に化けるだけだった。本当に必要なのは通すことでは
    なく**統合**（名寄せ）である。ここでは GT を使って各分裂クラスタを
    「その多数派話者の席」へ寄せた場合の成績を出す。実装可能な上限であって
    達成値ではない（実際の名寄せは声紋で決めるので、これより下がる）。

    この上限が大きければ、名寄せは**精度と未確定を同時に改善する**打ち手＝
    トレードオフではない、と言える。門番の調整（誤帰属↔未確定の交換）とは
    性質が違うので分けて測る。
    """
    loaded = load_run(run)
    if loaded is None:
        return None
    utts, code_by_ms = loaded
    rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
    rows = [(u, c) for u, c in rows if c in GT_CODES
            and not _BACKCHANNEL_RE.match(str(u["_text"]).strip())]
    if not rows or not any(u.get("final_key") is not None for u, _ in rows):
        return None

    def _score(final_of) -> dict:
        pairs = [(final_of(u), c) for u, c in rows]
        _a, mapping = _gtlib.best_mapping(pairs, GT_CODES, unsure=UNSURE_SPEAKER)
        n = len(pairs)
        ok = sum(1 for f, c in pairs if mapping.get(f) == c)
        uns = sum(1 for f, _ in pairs if f == UNSURE_SPEAKER)
        return {"acc": ok / n, "unsure": uns / n,
                "wrong": (n - ok - uns) / n, "n": n}

    def _cur(u: dict) -> str:
        return str(u["final_key"])

    # 席のあるキー（GTコードが付いたキー）と、分裂クラスタの多数派話者
    _a, seat_map = _gtlib.best_mapping([(_cur(u), c) for u, c in rows],
                                       GT_CODES, unsure=UNSURE_SPEAKER)
    seat_of_code = {v: k for k, v in seat_map.items() if v}
    frag: dict[str, Counter] = {}
    for u, code in rows:
        if _cur(u) != UNSURE_SPEAKER or str(u.get("key")) == UNSURE_SPEAKER:
            continue
        frag.setdefault(str(u["key"]), Counter())[code] += 1
    merged = {k: seat_of_code.get(cc.most_common(1)[0][0])
              for k, cc in frag.items()}

    def _merge(u: dict) -> str:
        cur = _cur(u)
        if cur != UNSURE_SPEAKER:
            return cur
        return merged.get(str(u.get("key"))) or cur

    n_pure = sum(1 for cc in frag.values()
                 if cc.most_common(1)[0][1] / sum(cc.values()) >= 0.9)
    return {"run": run, "before": _score(_cur), "after": _score(_merge),
            "frags": len(frag), "pure": n_pure}


def endorse_table(runs: list[str], kinds: set[str]) -> dict:
    """kind ごとに「裏付けの有無 × 結末」を数える（§18.8 が採否を決めた表）.

    門番を新しい kind に広げてよいかは、成績の増減より先に**この表**で決まる。
    裏付けありが正解に偏り、なしが誤帰属に偏っていれば弁別子として働く。
    偏りが無ければ、それは門番ではなく**全遮断**であり、§18.8 が
    「正解回収を失いすぎる」として退けた選択肢と同じものになる。

    ``name`` が一度も記録されない kind では裏付けが原理的に成立しないので、
    ``名前記録なし`` の件数も併せて出す（門番の形をした全遮断を見抜くため）。
    """
    tab: Counter = Counter()
    no_name: Counter = Counter()
    for run in runs:
        loaded = load_run(run)
        if loaded is None:
            continue
        utts, code_by_ms = loaded
        rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
        rows = [(u, c) for u, c in rows if c in GT_CODES
                and not _BACKCHANNEL_RE.match(str(u["_text"]).strip())]
        if not rows:
            continue

        def _cur(u: dict) -> str:
            return str(u["final_key"] if u.get("final_key") is not None
                       else u.get("key"))
        _a, mapping = _gtlib.best_mapping([(_cur(u), c) for u, c in rows],
                                          GT_CODES, unsure=UNSURE_SPEAKER)
        for u, code in rows:
            kind = u.get("kind")
            if kind not in kinds:
                continue
            if not u.get("name"):
                no_name[kind] += 1
            final = _cur(u)
            out = ("未確定" if final == UNSURE_SPEAKER
                   else "正解" if mapping.get(final) == code else "誤帰属")
            tab[(kind, endorsed(u), out)] += 1
    return {"tab": tab, "no_name": no_name}


def discover() -> list[str]:
    out = []
    for p in sorted((ROOT / "eval").glob("gt_*.json")):
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        s = g.get("session")
        if s and (ROOT / "transcripts" / f"{s}.diag.jsonl").exists() \
                and (ROOT / "transcripts" / f"{s}.turns.jsonl").exists():
            out.append(s)
    return out


def _print_endorse_table(runs: list[str], kinds: set[str]) -> None:
    """裏付けの弁別力を先に出す（成績の増減より前に見るべき表）."""
    res = endorse_table(runs, kinds)
    tab, no_name = res["tab"], res["no_name"]
    print("# 裏付け（声紋1位候補の一致 + sim≥閾値）の弁別力")
    print(f"{'kind':<12}{'裏付け':<8}{'正解':>6}{'誤帰属':>7}{'未確定':>7}{'正解率':>8}")
    for kind in sorted(kinds):
        for ok in (True, False):
            c = [tab[(kind, ok, x)] for x in ("正解", "誤帰属", "未確定")]
            if not any(c):
                continue
            decided = c[0] + c[1]
            rate = f"{c[0] / decided:.0%}" if decided else "—"
            print(f"{kind:<12}{'あり' if ok else 'なし':<8}"
                  f"{c[0]:>6}{c[1]:>7}{c[2]:>7}{rate:>8}")
        n_total = sum(tab[(kind, ok, x)] for ok in (True, False)
                      for x in ("正解", "誤帰属", "未確定"))
        if n_total and no_name[kind] == n_total:
            print(f"  ※ {kind} は name が一度も記録されない（{n_total}件全部）。"
                  "裏付けは原理的に成立せず、門番ではなく全遮断になる")


def _print_cf_mean(cfs: list[dict], label: str) -> None:
    n = len(cfs)
    m = {f"{w}_{k}": sum(c[w][k] for c in cfs) / n
         for w in ("before", "after") for k in ("acc", "wrong", "unsure")}
    print(f"{label:<20}{m['before_acc']:>8.1%}→{m['after_acc']:<8.1%}"
          f"{m['before_wrong']:>9.1%}→{m['after_wrong']:<9.1%}"
          f"{m['before_unsure']:>9.1%}→{m['after_unsure']:<9.1%}")
    # 差分は「点」＝パーセントポイント。割合のまま引くと 100 倍ずれる
    d_acc = (m["after_acc"] - m["before_acc"]) * 100
    d_wrong = (m["after_wrong"] - m["before_wrong"]) * 100
    d_unsure = (m["after_unsure"] - m["before_unsure"]) * 100
    line = (f"  {label}: 正解 {d_acc:+.1f}pt / 誤帰属 {d_wrong:+.1f}pt"
            f" / 未確定 {d_unsure:+.1f}pt")
    if d_wrong < 0:
        line += f" — 誤帰属1点あたり正解 {-d_acc / -d_wrong:.2f}点"
    print(line)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--detail", action="store_true",
                   help="誤帰属の声紋kind別内訳も出す")
    p.add_argument("--gate-kinds", default=None, metavar="KIND,KIND",
                   help="不純回収門番（§18.8）をこの kind にも適用した反実仮想を出す"
                        "（例: ラベル継続,蓄積中）")
    p.add_argument("--split", type=int, default=0, metavar="N",
                   help="先頭N本を開発セット・残りを検証セットとして分けて集計"
                        "（§18.8 と同じ作法。ランは名前順＝時刻順）")
    p.add_argument("--seat-recovery", action="store_true",
                   help="席落ち（constrain で落ちた分）を回収したときの成績。"
                        "門番を緩める打ち手と違い、精度とのトレードオフではない")
    p.add_argument("--merge-ceiling", action="store_true",
                   help="分裂クラスタを多数派話者の席へ完全に統合できた場合の上限。"
                        "精度と未確定を同時に改善しうる打ち手かを判定する")
    p.add_argument("--prefix", default=None,
                   help="この文字列で始まるランだけを対象にする"
                        "（例: 2026-07-20 で新コードの記録ランに限定）")
    args = p.parse_args(argv)
    runs = discover() if args.all else ([args.run] if args.run else [])
    if args.prefix:
        runs = [r for r in runs if r.startswith(args.prefix)]
    if not runs:
        raise SystemExit("# --run か --all を指定")
    if args.merge_ceiling:
        rs = [r for r in (merge_ceiling(x) for x in runs) if r]
        if not rs:
            raise SystemExit("# final_key を持つラン（2026-07-14以降）が必要")
        print("# クラスタ名寄せの上限: 分裂クラスタを多数派話者の席へ完全に統合")
        print("  （GT を使った上限。実際は声紋で寄せるのでこれより下がる）")
        print(f"{'run':<20}{'分裂':>5}{'うち純':>7}{'正解(前→後)':>18}"
              f"{'誤帰属(前→後)':>20}{'未確定(前→後)':>20}")
        for c in rs:
            b, a = c["before"], c["after"]
            print(f"{c['run']:<20}{c['frags']:>5}{c['pure']:>7}"
                  f"{b['acc']:>8.1%}→{a['acc']:<8.1%}"
                  f"{b['wrong']:>9.1%}→{a['wrong']:<9.1%}"
                  f"{b['unsure']:>9.1%}→{a['unsure']:<9.1%}")
        if len(rs) > 1:
            _print_cf_mean(rs, "平均")
        return
    if args.seat_recovery:
        rs = [r for r in (seat_recovery(x) for x in runs) if r]
        if not rs:
            raise SystemExit("# final_key を持つラン（2026-07-14以降）が必要")
        print("# 席落ちの回収: constrain 前のキーをそのまま採ったら何が起きるか")
        print("  （門番を緩める打ち手と違い、判定は変えない。席の帳簿の問題）")
        print(f"{'run':<20}{'落ちた数':>8}{'正解(前→後)':>18}"
              f"{'誤帰属(前→後)':>20}{'未確定(前→後)':>20}")
        for c in rs:
            b, a = c["before"], c["after"]
            print(f"{c['run']:<20}{c['dropped']:>8}"
                  f"{b['acc']:>8.1%}→{a['acc']:<8.1%}"
                  f"{b['wrong']:>9.1%}→{a['wrong']:<9.1%}"
                  f"{b['unsure']:>9.1%}→{a['unsure']:<9.1%}")
        if len(rs) > 1:
            _print_cf_mean(rs, "平均")
        return
    if args.gate_kinds:
        kinds = {k.strip() for k in args.gate_kinds.split(",") if k.strip()}
        _print_endorse_table(runs, kinds)
        cfs = [r for r in (counterfactual(x, kinds) for x in runs) if r]
        print(f"\n# 反実仮想: 不純回収門番を {sorted(kinds)} にも適用")
        print(f"{'run':<20}{'実質(前→後)':>18}{'誤帰属(前→後)':>20}"
              f"{'未確定(前→後)':>20}")
        for c in cfs:
            b, a = c["before"], c["after"]
            print(f"{c['run']:<20}{b['acc']:>8.1%}→{a['acc']:<8.1%}"
                  f"{b['wrong']:>9.1%}→{a['wrong']:<9.1%}"
                  f"{b['unsure']:>9.1%}→{a['unsure']:<9.1%}")
        if len(cfs) > 1:
            if args.split and 0 < args.split < len(cfs):
                _print_cf_mean(cfs[:args.split], f"開発{args.split}本平均")
                _print_cf_mean(cfs[args.split:],
                               f"検証{len(cfs) - args.split}本平均")
            else:
                _print_cf_mean(cfs, "平均")
        return
    results = [r for r in (decompose(x) for x in runs) if r]
    if not results:
        raise SystemExit("# 分解できるラン（GT付き）が無い")
    head = ["正解", "誤帰属", "席落ち", "ラベル不純", "重なり", "継続不可", "蓄積待ち"]
    keys = ["正解", "誤帰属", "未確定/席落ち", "未確定/ラベル不純",
            "未確定/重なり", "未確定/継続不可", "未確定/蓄積待ち"]
    print("# 帰属の損失分解（相槌を除く実質発話が分母）")
    print(f"{'run':<20}{'n':>5}" + "".join(f"{h:>9}" for h in head))
    for r in results:
        mark = "" if r["has_final"] else "  ※final_key無し(席落ち分離不可)"
        print(f"{r['run']:<20}{r['n']:>5}"
              + "".join(f"{r['share'][k]:>8.1%}" for k in keys) + mark)
    if len(results) > 1:
        n = len(results)
        print(f"{'平均':<20}{'':>5}"
              + "".join(f"{sum(x['share'][k] for x in results) / n:>8.1%}"
                        for k in keys))
    if args.detail:
        print("\n== 誤帰属の内訳（声紋層の kind 別） ==")
        total: Counter = Counter()
        for r in results:
            total.update(r["wrong_by_kind"])
        s = sum(total.values()) or 1
        for k, v in total.most_common():
            print(f"  {k or '(なし)':<12} {v:5d} ({v / s:5.1%})")
    print("\n読み方: 各列は「その原因を完全に解消したら最大で何pt返るか」の上限。")
    print("        全部が正解になる前提なので実際の伸びしろはこれより小さい。")
    print("        席落ち＝constrain で落ちた（上限設定・二重帳簿）")
    print("        ラベル不純＝Soniox の話者混載で声紋層が棄権")
    print("        蓄積待ち＝声紋もクラスタも未成熟（ヒステリシス未達等）")


if __name__ == "__main__":
    main()
