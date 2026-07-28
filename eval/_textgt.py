"""正解を「時間の重なり」ではなく「文章の一致」で割り当てる.

**なぜ要るのか**: 発話の区切りはランごとに変わるので、正解（コーパス側の
発話）とは1対1で対応しない。これまでは時間の重なりで割り当て、8割を一人が
占めるときだけ正解としていた。ところが3人の会話では、長い発話ほど笑いや
相づちが重なって主たる話者の取り分が薄まる——実測で、誰かが喋っている時間の
20%は2人以上が重なっている。結果、**採点できるのは全体の6割弱**になり、
しかも外れるのは「重なりが多い＝難しい場面」に偏っていた。難しい所を外して
採点していたことになる。

文字起こしが正確なら、文章そのものを突き合わせるほうが本質に近い。
「この一文は誰のものか」は、重なっていても一意に決まるからである。実測では
採点できる範囲が 59% → 84% に広がり、新たに測れた86件は正解47/誤帰属9/
未確定30——現行が取りこぼしていたのは、やはり成績の悪い側だった。

対応が付かないのは平均0.9秒の相づち（「おお。」「ね。」）で、同じ語が何度も
出るため文章では区別できない。ここは時間でも文章でも決まらないので、採点
対象外のまま残す（相づちの帰属は元々の関心事ではない）。
"""
from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402

# 候補にする時間の窓。区切りのずれを吸収する幅で、これ以上広げると
# 同じ相づちが遠くから拾われる。
WINDOW_MS = 3000
# 「対応する文がある」と認める一致度。0.45 は、言い直し・助詞の欠落程度は
# 通し、別の文は通さない線（実データで確認）。
MIN_SIM = 0.45
# 一致度から引く、時間の隔たりのペナルティ（秒あたり）。同じ語が何度も出る
# 相づちを、近いほうへ寄せるための重み。
GAP_PENALTY = 0.05

_BRACKET = re.compile(r"[<（(][^<>（）()]*[>）)]")
_PUNCT = re.compile(r"[、。？?！!・:：\s「」【】…ー―—]")


def normalize(text: str) -> str:
    """コーパスの記号と句読点を落として、比べられる形にする.

    コーパスは `(F_えーっと)` `(W_誤り|正しい)` `<笑>` のような記号を含む。
    括弧は入れ子になるので、変化しなくなるまで内側から剥がす。
    """
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\(W_[^|()]*\|([^()]*)\)", r"\1", text)   # (W_誤|正) → 正
        text = re.sub(r"\([A-Z]_([^()]*)\)", r"\1", text)        # (F_…) → 中身
        text = _BRACKET.sub("", text)
    return _PUNCT.sub("", text)


def source_of(run: str) -> str | None:
    """そのランの正解がどのコーパスから来たか（`transplanted_from`）."""
    p = ROOT / "eval" / f"gt_{run}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("transplanted_from")


def _corpus_turns(source: str) -> list[dict] | None:
    """コーパス側の (時刻つき発話 + 正解話者)。正解の付いた発話だけ返す."""
    gt_path = ROOT / "eval" / f"gt_{source}.json"
    turns_path = ROOT / "transcripts" / f"{source}.turns.jsonl"
    if not gt_path.exists() or not turns_path.exists():
        return None
    labels = json.loads(gt_path.read_text(encoding="utf-8")).get("labels") or {}
    out = []
    for t in _gtlib.read_jsonl(turns_path):
        code = labels.get(str(t["turn_id"]))
        if code in ("S1", "S2", "S3"):
            out.append({"ms": t["ms"], "end_ms": t["end_ms"], "code": code,
                        "norm": normalize(str(t.get("text", "")))})
    return out or None


def codes_by_ms(run: str, utterances: list[dict], *,
                text_key: str = "_text") -> dict[int, str] | None:
    """ランの発話それぞれに、文章の一致で正解コードを割り当てる.

    `utterances` は ``ms`` / ``end`` / テキストを持つ辞書の列（diag でも turns
    でも良い）。戻り値は ms -> "S1"|"S2"|"S3"。対応が付かない発話は入らない
    （＝採点対象外）。
    """
    source = source_of(run)
    corpus = _corpus_turns(source) if source else None
    if not corpus:
        return None
    out: dict[int, str] = {}
    for u in utterances:
        ms = int(u["ms"])
        end = int(u.get("end") or u.get("end_ms") or ms)
        text = normalize(str(u.get(text_key) or u.get("text") or ""))
        if not text:
            continue
        best_score, best_code = 0.0, None
        for c in corpus:
            if c["end_ms"] < ms - WINDOW_MS or c["ms"] > end + WINDOW_MS:
                continue
            gap = max(0, max(ms - c["end_ms"], c["ms"] - end)) / 1000.0
            score = SequenceMatcher(None, text, c["norm"]).ratio() - GAP_PENALTY * gap
            if score > best_score:
                best_score, best_code = score, c["code"]
        if best_code is not None and best_score >= MIN_SIM:
            out[ms] = best_code
    return out
