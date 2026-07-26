#!/usr/bin/env python3
"""文字起こし（テキスト）の精度を測る — CER / 取りこぼし率.

**このリポジトリで唯一の、実会話に対する文字起こし精度の測定**（2026-07-25 新設）。
これまで eval/ の採点はすべて「話者ラベルが正しいか」で、テキストの正確さは
合成テストセット（scripts/score_overlap_test.py の difflib 類似度）でしか
測られていなかった。「精度よく文字起こしできる」が製品の目的である以上、
そこに改善のループが無いのは穴だった。

材料は**既にリポジトリにある**（新規録音もAPIコストも不要）:

  参照: transcripts/<conv>.turns.jsonl
        eval/prep_chiba.py が Chiba3Party の形態論CSV（時刻つき）から生成した
        GT定義セッション。話者・時刻・**本文**を持つ
  仮説: transcripts/<run>.turns.jsonl
        同じ会話をシステムに流したときの出力
  対応: eval/gt_<run>.json の ``transplanted_from``

指標:

  CER  文字誤り率 = レーベンシュタイン距離 / 参照文字数。日本語は語境界が
       曖昧なので WER ではなく CER を使う（形態素解析器に依存しないため
       再現性も高い）
  取りこぼし率  参照にあってシステムが1文字も出さなかった時間帯の割合。
       CER と分けるのは、「認識を間違えた」と「そもそも拾えていない」が
       別の問題（前者はモデル、後者は VAD・区切り・重なり）だから

転記記号の正規化（Chiba は CSJ 系の転記規則）:

  (F_あのね)  フィラー      → 中身を残す（実際に発話されている）
  (D_ア)      言い直し断片   → 中身を残す
  (I_うん)    感動詞・応答詞 → 中身を残す
  <笑> <咳>   非言語音       → 落とす（音声であって言語ではない）
  (1.547)     ポーズ秒数     → 落とす
  :           母音延伸       → 落とす（「あー」を「あ:」と書く記法）
  (?_...)     聞き取り不能   → 中身を残す（参照側の不確かさは分母に含める）

フィラーを残すかは判断が分かれるので `--no-filler` で除いた値も出せる。
既定は「残す」＝話された音をすべて分母に入れる厳しい側。

使い方:
    uv run python eval/score_transcription.py --run 2026-07-20_1723
    uv run python eval/score_transcription.py --all      # 対応の付く全ラン
    uv run python eval/score_transcription.py --all --no-filler
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402

# (X_中身) 形式のタグ。中身は実際に発話されているので既定では残す。
_TAG_RE = re.compile(r"\(([A-Za-z?])_([^()]*)\)")
# <笑> <咳> 等の非言語音
_NONVERBAL_RE = re.compile(r"[<＜][^<>＜＞]*[>＞]")
# (1.547) のようなポーズ表記
_PAUSE_RE = re.compile(r"\(\d+(?:\.\d+)?\)")
# 記号・空白（両側から落として比較の土俵を揃える）
_PUNCT_RE = re.compile(r"[\s　:：、。,.!?！？「」『』（）()・…\-―ー～~"
                       r"・、。]+")


def normalize(text: str, *, keep_filler: bool = True) -> str:
    """比較用に転記記号を落とす（規則は本モジュールの docstring）."""
    t = unicodedata.normalize("NFKC", str(text))
    t = _NONVERBAL_RE.sub("", t)
    t = _PAUSE_RE.sub("", t)

    def _tag(m: re.Match) -> str:
        kind, body = m.group(1), m.group(2)
        if not keep_filler and kind.upper() in ("F", "D"):
            return ""      # フィラー・言い直し断片を落とす
        return body
    t = _TAG_RE.sub(_tag, t)
    t = _PUNCT_RE.sub("", t)
    return t


def cer(ref: str, hyp: str) -> tuple[int, int]:
    """(編集距離, 参照長) を返す（レーベンシュタイン距離・O(len(ref)*len(hyp))）."""
    if not ref:
        return len(hyp), 0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1,          # 削除
                           cur[j - 1] + 1,       # 挿入
                           prev[j - 1] + (rc != hc)))   # 置換
        prev = cur
    return prev[-1], len(ref)


def load_pairs(run: str) -> tuple[list[dict], list[dict], str]:
    """(参照ターン, システムターン, 会話名) を返す."""
    gt_path = ROOT / "eval" / f"gt_{run}.json"
    if not gt_path.exists():
        raise SystemExit(f"# {gt_path} が無い（対応表が引けない）")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    conv = gt.get("transplanted_from")
    if not conv:
        raise SystemExit(
            f"# gt_{run}.json に transplanted_from が無い。"
            "参照本文つきのコーパス由来ランでのみ測定できる")
    root = ROOT / "transcripts"
    ref = _gtlib.read_jsonl(root / f"{conv}.turns.jsonl")
    hyp = _gtlib.read_jsonl(root / f"{run}.turns.jsonl")
    return ref, hyp, conv


def missed_rate(ref: list[dict], hyp: list[dict]) -> float:
    """参照の発話時間のうち、システム出力が1つも重ならなかった割合."""
    spans = [(int(h["ms"]), int(h.get("end_ms") or h["ms"])) for h in hyp
             if h.get("text")]
    spans.sort()
    total = missed = 0
    for r in ref:
        if not str(r.get("text", "")).strip():
            continue
        a, b = int(r["ms"]), int(r.get("end_ms") or r["ms"])
        dur = max(0, b - a)
        total += dur
        if not any(min(b, y) - max(a, x) > 0 for x, y in spans):
            missed += dur
    return missed / total if total else 0.0


def _by_window(turns: list[dict], window_ms: int) -> dict[int, str]:
    """開始時刻の窓ごとに本文を連結する（両者を同じ規則で仕切る）."""
    out: dict[int, list[str]] = {}
    for t in turns:
        w = int(t["ms"]) // window_ms
        out.setdefault(w, []).append(str(t.get("text", "")))
    return {w: "".join(v) for w, v in out.items()}


def score_run(run: str, *, keep_filler: bool, window_sec: float = 0.0
              ) -> dict | None:
    """CER を計算する（既定は全文連結。``window_sec>0`` で時間窓ごと）.

    **既定が全文連結なのは実測で決めた。** 当初「参照は相槌を独立ターンとして
    挟むのに対しシステムは長くまとめるので、全文連結では順序ずれが誤差に化ける」
    と考えて開始時刻10秒窓を試したが、**CER は 32%→50% と悪化した**
    （chiba0132 は 16%→75%）。原因は逆で、システムのターンが40秒級に長く、
    1つのターンが複数の窓をまたぐため、参照だけが後続の窓に残って全削除に
    数えられる。窓で仕切ると「区切り方の違い」が「認識の誤り」に化ける。

    相槌の順序ずれは残るが、相槌は数文字と短く影響は小さい。区切りに依存しない
    全文連結のほうが「何を認識できたか」の推定として素直、というのが結論。
    ``--window`` は残してあるので、区切りの一致度を見たいときには使える。
    """
    ref, hyp, conv = load_pairs(run)
    if window_sec <= 0:
        ref_w = {0: "".join(str(r.get("text", "")) for r in ref)}
        hyp_w = {0: "".join(str(h.get("text", "")) for h in hyp)}
    else:
        window_ms = int(window_sec * 1000)
        ref_w = _by_window(ref, window_ms)
        hyp_w = _by_window(hyp, window_ms)
    dist = n = 0
    hyp_chars = 0
    for w in sorted(set(ref_w) | set(hyp_w)):
        r = normalize(ref_w.get(w, ""), keep_filler=keep_filler)
        h = normalize(hyp_w.get(w, ""), keep_filler=keep_filler)
        hyp_chars += len(h)
        if not r and not h:
            continue
        d, m = cer(r, h)
        dist += d
        n += m
    if not n:
        return None
    return {
        "run": run, "conv": conv,
        "cer": dist / n,
        "ref_chars": n, "hyp_chars": hyp_chars,
        "len_ratio": hyp_chars / n,
        "missed": missed_rate(ref, hyp),
    }


def discover_runs() -> list[str]:
    out = []
    for p in sorted((ROOT / "eval").glob("gt_*.json")):
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        conv = g.get("transplanted_from")
        run = g.get("session")
        if conv and run and (ROOT / "transcripts" / f"{conv}.turns.jsonl").exists() \
                and (ROOT / "transcripts" / f"{run}.turns.jsonl").exists():
            out.append(run)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=None, help="採点するラン（セッション名）")
    p.add_argument("--all", action="store_true", help="対応の付く全ランを採点")
    p.add_argument("--no-filler", action="store_true",
                   help="フィラー・言い直し断片を参照から除いて測る")
    p.add_argument("--window", type=float, default=0.0,
                   help="突き合わせの時間窓（秒）。既定0＝全文連結。"
                        "窓で仕切ると区切り方の違いが誤りに化けるため既定は0"
                        "（score_run の docstring に実測の経緯）")
    args = p.parse_args(argv)
    runs = discover_runs() if args.all else ([args.run] if args.run else [])
    if not runs:
        raise SystemExit("# --run か --all を指定（対応の付くランが無い場合も空）")
    keep = not args.no_filler
    how = "全文連結" if args.window <= 0 else f"{args.window:.0f}秒窓"
    print(f"# 文字起こし精度（CER・{'フィラー込み' if keep else 'フィラー除き'}"
          f"・{how}で突き合わせ）")
    print(f"{'run':<20}{'会話':<12}{'CER':>8}{'取りこぼし':>10}"
          f"{'長さ比':>8}{'参照文字':>9}")
    rows = []
    for run in runs:
        r = score_run(run, keep_filler=keep, window_sec=args.window)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['run']:<20}{r['conv']:<12}{r['cer']:>7.1%}"
              f"{r['missed']:>10.1%}{r['len_ratio']:>8.2f}{r['ref_chars']:>9}")
    if len(rows) > 1:
        n = len(rows)
        print(f"{'平均':<32}{sum(x['cer'] for x in rows) / n:>7.1%}"
              f"{sum(x['missed'] for x in rows) / n:>10.1%}"
              f"{sum(x['len_ratio'] for x in rows) / n:>8.2f}")
    print("\n読み方:")
    print("  CER        文字誤り率。参照の転記記号は正規化済み（docstring 参照）")
    print("  取りこぼし  参照の発話時間のうち、システムが1文字も出さなかった割合。")
    print("             CER と分けるのは「認識を間違えた」（モデルの問題）と")
    print("             「そもそも拾えていない」（VAD・区切り・重なりの問題）が")
    print("             別の打ち手になるため")
    print("  長さ比      システム出力文字数 ÷ 参照文字数。1.0 より大きく外れる")
    print("             ときは、CER の内訳が挿入寄りか削除寄りかの手がかり")


if __name__ == "__main__":
    main()
