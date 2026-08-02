"""名乗り判定のLLM化(§49.14)の事前評価——実装前に83ランで精度を測る.

二段構えの設計(ユーザー合意 2026-08-02):
  1. 緩い事前フィルタ(高再現・低精度)で候補発話を絞る
  2. 候補だけを安いLLMに「自己紹介か? 名前は?」と判定させる(バッチ)
本番組み込みは、この評価が厳格な正規表現(83ラン誤発火0・§49.11)と
同等以上と確認できてから。厳格版は --no-llm 時のフォールバックとして残す。

正解の扱い: 厳格な正規表現の検出(12件)を既知の正例とし、LLMだけが
正と言った発話は「新発見候補」として列挙する(自動では誤りと数えない。
正規表現が取りこぼしてきた形かもしれないため、目視レビューに回す)。

使い方(Mac・OPENAI_API_KEY 必要。数百件×バッチで数円程度):
  uv run python eval/nanori_llm.py
  uv run python eval/nanori_llm.py --model gpt-5.4-mini --limit 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from das.asr.live._nanori import detect_nanori  # noqa: E402

OPENAI_API = "https://api.openai.com/v1/chat/completions"

# 緩い事前フィルタ: 名乗りの動詞的な手がかり、または「の+短い語+です」。
# 高再現が目的で、精度はLLMに任せる。
_CAND = re.compile(
    r"(申します|といいます|と言います|でございます"
    r"|の\s*[一-龥ぁ-んァ-ヶーA-Za-z]{2,8}(です|でした))")

_PROMPT = """会議・番組の書き起こしの発話を番号付きで与えます。
各発話について、話者が**自分自身を名乗っている**(自己紹介している)かを判定してください。
- 他人の紹介(「続いては田中さんです」)、引用、呼びかけ、単なる「〜です」文は名乗りではない
- 他人を紹介してから自分も名乗る発話(「…の田中さんと、私、鈴木で…」)は、**自分の名前だけ**を抜き出す
- 音声認識の聞き取りが崩れていて氏名として不自然な場合は nanori=false にする
- 「私は賛成です」のような代名詞+意見は名乗りではない。代名詞(私/自分/僕等)を name にしない
- 名乗りなら、名乗った氏名(姓または姓名。所属・肩書きは含めない)を抜き出す
JSONの配列だけを返す: [{"i": 番号, "nanori": true/false, "name": "氏名またはnull"}]"""

# 氏名らしさの門: 日本語の姓名で8文字を超えることはまず無い。STTが崩れた
# 挨拶が名前扱いされるのを防ぐ(1回目の評価で「しんのじはうまへん」9文字が
# 名前として抽出された実測から)。
NAME_MAX_CHARS = 8
# 代名詞は氏名ではない(言語的事実。2周目でシミュレーション討論の
# 「私は賛成です」を名乗り判定し名前「私」を抽出した実測から)
PRONOUNS = frozenset({"私", "わたし", "わたくし", "自分", "僕", "ぼく",
                      "俺", "おれ", "当方", "こちら", "うち"})


def plausible_name(name) -> bool:
    """LLMが返した氏名が登録に値するか（空・長さ・代名詞の門。中身は問わない）."""
    n = str(name or "").strip()
    return bool(n) and len(n) <= NAME_MAX_CHARS and n not in PRONOUNS


def _chat(model: str, api_key: str, content: str, *, timeout: int = 60):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": _PROMPT},
                     {"role": "user", "content": content}],
        "max_completion_tokens": 4000,
    }).encode()
    req = urllib.request.Request(OPENAI_API, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    usage = resp.get("usage", {})
    return json.loads(text), usage


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL_FAST",
                                                     "gpt-5.4-mini"))
    p.add_argument("--limit", type=int, default=None,
                   help="候補をこの件数までに制限（素振り用）")
    p.add_argument("--batch", type=int, default=25)
    args = p.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")   # 他スクリプトと同じ流儀（make_debate_wav 等）
    except ImportError:
        pass
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が未設定です（.env にも見つかりません）")

    # 候補の収集
    cands = []   # (run, turn_id, text, regex_name)
    n_utt = 0
    for path in sorted(glob.glob(str(ROOT / "transcripts" / "*.turns.jsonl"))):
        run = Path(path).name.replace(".turns.jsonl", "")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            t = json.loads(line)
            tx = str(t.get("text") or "")
            if not tx or str(t.get("speaker")) == "ファシリテーター":
                continue
            n_utt += 1
            if _CAND.search(tx[:45]):
                cands.append((run, t.get("turn_id"), tx, detect_nanori(tx)))
    if args.limit:
        # 正例を必ず含めた上で先頭から切る
        pos = [c for c in cands if c[3]]
        neg = [c for c in cands if not c[3]][: max(0, args.limit - len(pos))]
        cands = pos + neg
    print(f"全発話 {n_utt} / 候補 {len(cands)}"
          f"（うち正規表現の正例 {sum(1 for c in cands if c[3])}）")

    # バッチ判定
    results = {}
    total_tokens = [0, 0]
    for i in range(0, len(cands), args.batch):
        chunk = cands[i:i + args.batch]
        content = "\n".join(f"{j}. {c[2][:160]}" for j, c in enumerate(chunk))
        try:
            arr, usage = _chat(args.model, api_key, content)
        except Exception as e:
            print(f"# バッチ{i//args.batch}失敗: {type(e).__name__}: {e}")
            continue
        total_tokens[0] += usage.get("prompt_tokens", 0)
        total_tokens[1] += usage.get("completion_tokens", 0)
        for row in arr:
            with_idx = i + int(row.get("i", -1))
            if 0 <= with_idx < len(cands):
                results[with_idx] = row
        print(f"  {min(i+args.batch, len(cands))}/{len(cands)} 済", flush=True)

    # 突き合わせ
    miss, extra, agree = [], [], 0
    for idx, (run, tid, tx, rx_name) in enumerate(cands):
        r = results.get(idx, {})
        llm_pos = bool(r.get("nanori")) and plausible_name(r.get("name"))
        if rx_name and llm_pos:
            agree += 1
        elif rx_name and not llm_pos:
            miss.append((run, tid, rx_name, tx[:40]))
        elif llm_pos and not rx_name:
            extra.append((run, tid, r.get("name"), tx[:40]))
    print()
    print(f"== 結果（model={args.model}）==")
    print(f"正規表現の正例をLLMも正と判定: {agree}/{agree + len(miss)}")
    for m in miss:
        print("  LLMが見逃した正例:", m)
    print(f"LLMだけが正と判定（新発見候補・要目視レビュー）: {len(extra)}件")
    for e in extra[:30]:
        print("   ", e)
    print(f"トークン: 入力{total_tokens[0]} / 出力{total_tokens[1]}")
    out = ROOT / "eval" / "_nanori_llm_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "agree": agree,
                   "miss": miss, "extra": extra,
                   "candidates": len(cands), "tokens": total_tokens},
                  f, ensure_ascii=False, indent=1)
    print(f"詳細: {out}")


if __name__ == "__main__":
    main()
