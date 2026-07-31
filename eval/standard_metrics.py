"""標準指標（DER / cpCER）で千葉12本を採点する——外部比較可能性のための測定.

これまでの評価は独自指標（文字ベースの帰属正解率、`_pipeline.score`）に
閉じていた（docs/research §8 の弱点）。ここでは論文で使われる標準の
道具・定義で同じランを測る:

- **DER** (Diarization Error Rate): `pyannote.metrics`。正解はコーパスの
  形態素時刻（Morph CSV）から話者ごとに区間化したタイムライン、仮説は
  本システムの発話区間＋最終話者。collar 0.5秒（±0.25s、慣例値）と 0 の
  両方、重なり込み/除きの両方を出す
- **cpCER** (concatenated minimum-permutation CER): `meeteval` の cpWER 実装に
  文字単位で流す。日本語のSA-ASRはCERで報告するのが通例（CSJ/CHiME系）。
  正解はコーパス転記（記号は `_textgt.normalize` で除去）、仮説は本システム
  の議事録テキスト。**話者の対応は cp（最小置換）が取る**ので、帰属の誤りは
  文字誤りとして跳ね返る——STT品質と帰属品質の複合指標である点に注意

前提: 正解転記は data/chiba/Chiba3Party/Morph/（コーパス由来・git非追跡）。
ランは transcripts/chibaXX32.*（再生ラン）。評価範囲は各ランの録音長に切る。

使い方: uv run python eval/standard_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import _textgt  # noqa: E402
from meeteval.wer.wer.cp import cp_word_error_rate  # noqa: E402
from prep_chiba import load_tokens  # noqa: E402
from pyannote.core import Annotation, Segment  # noqa: E402
from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: E402

# システム出力のラン → 正解コーパス会話（gt の transplanted_from）。
# chibaXX32.turns.jsonl 自体はコーパス正解の turns 形式なので仮説には使えない
# （最初の実装で自分自身と比較して DER 0% を出した——即座に疑って修正）。
RUNS = None  # main() で eval/gt_*.json から動的に構築
UNSURE_DISPLAY = "未確定"   # turns.jsonl は表示名を持つ（内部キー "?" ではない）
MERGE_GAP_SEC = 0.5   # 正解の単語トークンを発話区間へ併合する間隙（慣例的な値）


def _load_pairs() -> list[tuple[str, str]]:
    """(システムのラン, 正解の会話ID) を gt の transplanted_from から集める."""
    import glob
    pairs = []
    for p in sorted(glob.glob(str(ROOT / "eval" / "gt_2026-*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        conv = d.get("transplanted_from")
        run = d.get("session")
        if conv and run and (ROOT / "transcripts" / f"{run}.turns.jsonl").exists():
            pairs.append((run, conv))
    return pairs


def _run_duration_sec(run: str, conv: str) -> float:
    """評価範囲（秒）。正解トークンと仮説turnsの終端の重なる範囲で評価する.

    wav はクラウドに無い（記録のみ）。ランが会話全体を覆っていれば
    コーパス終端≒仮説終端になる。途中で終わったランを正解全長で採点すると
    後半が丸ごと miss になるため、短い側に切る（差が3%超なら注意を出す）。
    """
    ref_end = max(e for _s, e, _w, _t in load_tokens(conv))
    hyp_end = 0.0
    with open(ROOT / "transcripts" / f"{run}.turns.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t.get("end_ms") is not None:
                hyp_end = max(hyp_end, t["end_ms"] / 1000)
    if hyp_end < ref_end * 0.97:
        print(f"# 注意: {run} は仮説が {hyp_end:.0f}s で終わる"
              f"（正解 {ref_end:.0f}s）。短い側で評価", flush=True)
    return min(ref_end, hyp_end)


def _reference(conv: str, until_sec: float):
    """コーパス転記から (話者別タイムライン, 話者別テキスト) を作る."""
    ann = Annotation()
    texts: dict[str, list[str]] = {}
    cur: dict[str, list] = {}   # who -> [start, end]
    _flat: list[tuple[float, str]] = []
    for s, e, who, text in load_tokens(conv):
        if s >= until_sec:
            continue
        e = min(e, until_sec)
        norm = _textgt.normalize(text)
        texts.setdefault(who, []).append(norm)
        _flat.append((s, norm))
        c = cur.get(who)
        if c is not None and s - c[1] <= MERGE_GAP_SEC:
            c[1] = max(c[1], e)
        else:
            if c is not None:
                ann[Segment(c[0], c[1]), f"ref-{who}-{len(texts[who])}"] = who
            cur[who] = [s, e]
    for who, c in cur.items():
        ann[Segment(c[0], c[1]), f"ref-{who}-tail"] = who
    ordered = "".join(x for _s, x in sorted(_flat))
    return ann, {w: "".join(t) for w, t in texts.items()}, ordered


def _hypothesis(run: str, *, drop_unsure: bool = False):
    """本システムの turns.jsonl から (タイムライン, 話者別テキスト) を作る.

    drop_unsure=True は未確定の発話を仮説から外す変種。未確定は独立話者と
    して confusion に数えられるため、これを外すと「確定した発話の取り違え」
    と「未確定に逃がした分（missへ移る）」を分離できる。
    """
    ann = Annotation()
    texts: dict[str, list[str]] = {}
    _flat: list[tuple[int, str]] = []
    with open(ROOT / "transcripts" / f"{run}.turns.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t.get("ms") is None or t.get("end_ms") is None:
                continue
            sp = str(t["speaker"])
            if sp == "ファシリテーター":
                continue
            if drop_unsure and sp == UNSURE_DISPLAY:
                continue
            ann[Segment(t["ms"] / 1000, t["end_ms"] / 1000), f"h{t['turn_id']}"] = sp
            norm = _textgt.normalize(str(t.get("text") or ""))
            texts.setdefault(sp, []).append(norm)
            _flat.append((t["ms"], norm))
    ordered = "".join(x for _m, x in sorted(_flat))
    return ann, {s: "".join(x) for s, x in texts.items()}, ordered


def _spaced(s: str) -> str:
    """文字単位のCERにするため1文字ずつ空白区切りにする."""
    return " ".join(s)


def main() -> None:
    der_std = DiarizationErrorRate(collar=0.5)              # ±0.25s（慣例）
    der_strict = DiarizationErrorRate(collar=0.0)
    der_no_ov = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    der_decided = DiarizationErrorRate(collar=0.5)          # 未確定を外した変種
    cp_err = cp_len = 0
    asr_err = asr_len = 0                                   # 話者無視のCER（STT品質）
    pairs = _load_pairs()
    print(f"{'run':<18}{'正解':<11}{'DER(c=.25)':>11}{'DER(c=0)':>10}{'DER(重なり除く)':>14}{'cpCER':>8}")
    for run, conv in pairs:
        dur = _run_duration_sec(run, conv)
        ref_ann, ref_txt, ref_ordered = _reference(conv, dur)
        hyp_ann, hyp_txt, hyp_ordered = _hypothesis(run)
        uem = Segment(0, dur)
        d1 = der_std(ref_ann, hyp_ann, uem=uem)
        d2 = der_strict(ref_ann, hyp_ann, uem=uem)
        d3 = der_no_ov(ref_ann, hyp_ann, uem=uem)
        cp = cp_word_error_rate({k: _spaced(v) for k, v in ref_txt.items()},
                                {k: _spaced(v) for k, v in hyp_txt.items()})
        cp_err += cp.errors
        cp_len += cp.length
        hyp_dec, _dt, _do = _hypothesis(run, drop_unsure=True)
        der_decided(ref_ann, hyp_dec, uem=uem)
        # 話者を無視した純粋なSTT側のCER（両者とも時刻順に連結した全文どうし）
        asr = cp_word_error_rate({"all": _spaced(ref_ordered)},
                                 {"all": _spaced(hyp_ordered)})
        asr_err += asr.errors
        asr_len += asr.length
        # 未確定は独立クラスタとして扱われる（最適対応から外れ confusion になる）
        print(f"{run:<18}{conv:<11}{d1:>11.1%}{d2:>10.1%}{d3:>14.1%}{cp.error_rate:>8.1%}")
    print(f"\n## 集計（{len(pairs)}本・時間/文字の重み付き）")
    print(f"DER (collar±0.25s, 重なり込み): {abs(der_std):.1%}")
    print(f"DER (collar 0,     重なり込み): {abs(der_strict):.1%}")
    print(f"DER (collar±0.25s, 重なり除く): {abs(der_no_ov):.1%}")
    # 成分分解: DER = miss（正解に音声があるのに仮説に無い。重なりの取り逃し）
    #          + false alarm（仮説だけに音声。長い発話の間や余韻）
    #          + confusion（話者の取り違え。帰属の誤りに対応するのはここ）
    r = der_std[:]
    tot = r["total"] or 1
    print(f"  内訳(c=.25): miss {r['missed detection']/tot:.1%} / "
          f"false alarm {r['false alarm']/tot:.1%} / "
          f"confusion {r['confusion']/tot:.1%}")
    rd = der_decided[:]
    td = rd["total"] or 1
    print(f"  未確定を仮説から外すと: miss {rd['missed detection']/td:.1%} / "
          f"false alarm {rd['false alarm']/td:.1%} / "
          f"confusion {rd['confusion']/td:.1%}"
          f"  ←確定発話だけの取り違えはこの confusion")
    print(f"cpCER: {cp_err / cp_len:.1%}  （STT誤り＋帰属誤りの複合。話者対応は最小置換）")
    print(f"  話者無視のCER（STT品質のみ）: {asr_err / asr_len:.1%}"
          f" → 帰属による上乗せ ≈ {cp_err / cp_len - asr_err / asr_len:+.1%}")
    print(f"参考: 未確定ラベル({UNSURE_DISPLAY})は独立話者として計上（不利側の扱い）")


if __name__ == "__main__":
    main()
