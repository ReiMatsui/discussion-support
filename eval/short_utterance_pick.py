#!/usr/bin/env python3
"""1秒未満の席選びを、新しい測り方で測り直す（残る誤帰属の8割がここ）.

**なぜ測り直すのか**: 同じ2案（問い合わせ音声に文脈を足す／寄せ先の選び方を
変える）は以前も測って「効果なし」と却下した。しかしその判定は**時間の
重なりで正解を当てた数字**に基づいていた。その当て方は重なりの多い場面を
丸ごと採点から落とし、落ちていたのはまさに短い発話である（§34）。**効くか
どうかを見たい層が、分母から抜けていた**。判定そのものが無効なので、文章の
一致で当て直して測る。

内訳（§35）: 全誤帰属226件のうち184件（81%）が1秒未満で、そのうち179件は
「席の音声で決め直す」経路（ラベル不純・ラベル継続）を通っている。席の割当ては
閉集合の rank-1 で棄権が無いため、0.5秒の「ほう。」からでも必ず1席を選ぶ。

比べる案は2種類ある。

**問い合わせ音声**（席と比べる音声を何にするか。埋め込みの計算が増える）

    own      いまの実装（その発話の音声だけ）
    label    同じ Soniox ラベルの過去の音声を前に足す。ラベルは複数人を混載
             しうる（それが「ラベル不純」の意味）ので、その汚染が効くかを見る
    key      同じ上流キー（@diar:N 等）の過去の音声を足す
    label_w  label と同じだが直近30秒以内に限る（古い対応の持ち越しを断つ）

  足す量は10秒と2秒の両方を見る。0.5秒の発話に10秒を足せば足した側が
  埋め込みを支配するので、「文脈が効くか」は量と切り離しては答えられない。

**寄せ先の選び方**（同じ声紋のまま、選び方だけ変える。計算は増えない）

    rank1    いまの実装（1位を無条件に採る）
    margin   1位と2位の差が小さければ**未確定に落とす**
    floor    1位の類似そのものが低ければ未確定に落とす

margin と floor は正解率を下げる方向にも働く。それでも見るのは「誤帰属より
未確定を優先」という方針があるためで、誤帰属が減った分だけ未確定が増えるなら
採る価値がある。

いずれも過去の音声しか使わないので遅延は増えない。GT は採点にしか使わない。
新規録音も STT の再課金も不要。

使い方:
    uv run python eval/short_utterance_pick.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import SR, UNSURE_SPEAKER  # noqa: E402

KEEP = 40               # 履歴として持つ発話数（それ以上は古い方から捨てる）

# 問い合わせ音声の作り方: 名前 -> (どこから足すか, 足す秒数, 遡る上限ms)
# 10秒と2秒の両方を見るのは、0.5秒の発話に10秒を足すと**足した側が埋め込みを
# 支配する**ため。文脈の量そのものが効き方を決めるので、量を変えて比べる。
QUERY_SPECS: dict[str, tuple[str, float, int]] = {
    "": ("none", 0.0, 0),
    "label": ("label", 10.0, 0),
    "label2": ("label", 2.0, 0),
    "label_w": ("label", 10.0, 30_000),
    "key": ("key", 10.0, 0),
    "key2": ("key", 2.0, 0),
}


# ------------------------------------------------------ 問い合わせ音声


def _tail(chunks: list[np.ndarray], sec: float) -> np.ndarray:
    """直近 sec 秒ぶんを新しい方から取り、時間順に戻して連結する."""
    budget = int(sec * SR)
    out = []
    for a in reversed(chunks):
        if budget <= 0:
            break
        out.append(a[-budget:] if a.size > budget else a)
        budget -= min(a.size, budget)
    return np.concatenate(out[::-1]) if out else np.zeros(0, dtype=np.float32)


class Context:
    """発話ごとに全案ぶんの問い合わせ音声を作る（材料は過去の発話だけ）.

    記録を判定の**後**に行うのは、その発話自身を文脈に含めないため——含めると
    同じ音声が二重に効き、文脈の効果を過大に見積もる。
    """

    def __init__(self) -> None:
        self.by_label: dict[str, list[np.ndarray]] = {}
        self.by_label_ms: dict[str, list[int]] = {}
        self.by_key: dict[str, list[np.ndarray]] = {}

    def __call__(self, u: dict, wav: np.ndarray,
                 revisable: bool) -> dict[str, np.ndarray]:
        lab, key, ms = str(u.get("label")), str(u.get("key")), int(u["ms"])
        out = {}
        if revisable and wav.size:
            for name in QUERY_SPECS:
                out[name] = self._build(name, wav, lab, key, ms)
        self._remember(wav, lab, key, ms)
        return out

    def _build(self, name, wav, lab, key, ms) -> np.ndarray:
        src, sec, window = QUERY_SPECS[name]
        if src == "none":
            return wav
        if src == "key":
            past = self.by_key.get(key, []) if key != UNSURE_SPEAKER else []
        else:
            past = self.by_label.get(lab, [])
            if window:
                past = [c for c, m in zip(past, self.by_label_ms.get(lab, []),
                                          strict=True) if ms - m <= window]
        return np.concatenate([_tail(past, sec), wav]) if past else wav

    def _remember(self, wav, lab, key, ms) -> None:
        if not wav.size:
            return
        self.by_label.setdefault(lab, []).append(wav)
        self.by_label_ms.setdefault(lab, []).append(ms)
        del self.by_label[lab][:-KEEP], self.by_label_ms[lab][:-KEEP]
        if key != UNSURE_SPEAKER:
            self.by_key.setdefault(key, []).append(wav)
            del self.by_key[key][:-KEEP]


# ------------------------------------------------------ 寄せ先の選び方


def _sims(emb, refs: dict) -> list[tuple[float, str]]:
    if emb is None or len(refs) < 2:
        return []
    return sorted(((float(np.dot(emb, v)), k) for k, v in refs.items()),
                  reverse=True)


def by_margin(delta: float):
    """1位と2位の差が delta 未満なら未確定に落とす選び方を作る."""
    def pick(emb, refs):
        s = _sims(emb, refs)
        if not s:
            return None
        return s[0][1] if s[0][0] - s[1][0] >= delta else UNSURE_SPEAKER
    return pick


def by_floor(theta: float):
    """1位の類似が theta 未満なら未確定に落とす選び方を作る."""
    def pick(emb, refs):
        s = _sims(emb, refs)
        if not s:
            return None
        return s[0][1] if s[0][0] >= theta else UNSURE_SPEAKER
    return pick


VARIANTS: list[tuple[str, str, object]] = [
    ("いまの実装", "", None),
    ("文脈: ラベル10秒", "label", None),
    ("文脈: ラベル2秒", "label2", None),
    ("文脈: ラベル10秒/30秒内", "label_w", None),
    ("文脈: 上流キー10秒", "key", None),
    ("文脈: 上流キー2秒", "key2", None),
    ("差 0.03未満は未確定", "", by_margin(0.03)),
    ("差 0.05未満は未確定", "", by_margin(0.05)),
    ("差 0.10未満は未確定", "", by_margin(0.10)),
    ("類似 0.30未満は未確定", "", by_floor(0.30)),
    ("類似 0.40未満は未確定", "", by_floor(0.40)),
]


# ---------------------------------------------------------------- 採点


def outcomes(data: dict, name: str, pick) -> list[dict]:
    """1ランを指定の案で採点し、発話ごとの結末を返す（対応づけはラン単位）."""
    steps = data["steps"]
    final = pipe.apply_schedule(steps, pick=pick, name=name)
    pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    out = []
    for f, st in zip(final, steps, strict=True):
        u = st["utt"]
        out.append({
            "dur_ms": max(0, int(u.get("end") or u["ms"]) - int(u["ms"])),
            "chars": len(str(u.get("_text") or "")),
            "outcome": ("未確定" if f == UNSURE_SPEAKER
                        else "正解" if m.get(f) == st["code"] else "誤帰属"),
        })
    return out


def summarize(rows: list[dict]) -> tuple[int, tuple, tuple]:
    """(件数, 件数の内訳, 文字数の内訳) を割合で返す."""
    n = len(rows) or 1
    w = sum(r["chars"] for r in rows) or 1
    cnt = tuple(sum(1 for r in rows if r["outcome"] == k) / n
                for k in ("正解", "誤帰属", "未確定"))
    chr_ = tuple(sum(r["chars"] for r in rows if r["outcome"] == k) / w
                 for k in ("正解", "誤帰属", "未確定"))
    return len(rows), cnt, chr_


def _table(label: str, rows_by_variant: dict[str, list[dict]],
           keep=lambda r: True) -> None:
    print(f"\n## {label}")
    print(f"{'案':<24}{'件数':>6}{'正解':>7}{'誤帰属':>7}{'未確定':>7}"
          f"{'  ':>2}{'文字':>7}{'正解':>7}{'誤帰属':>7}{'未確定':>7}")
    for name, _q, _p in VARIANTS:
        rows = [r for r in rows_by_variant[name] if keep(r)]
        if not rows:
            continue
        n, cnt, ch = summarize(rows)
        w = sum(r["chars"] for r in rows)
        print(f"{name:<24}{n:>6}{cnt[0]:>7.1%}{cnt[1]:>7.1%}{cnt[2]:>7.1%}"
              f"{'  ':>2}{w:>7}{ch[0]:>7.1%}{ch[1]:>7.1%}{ch[2]:>7.1%}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    p.add_argument("--split", type=int, default=5,
                   help="開発/検証に分ける本数（0で分けない）")
    args = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]

    # 1本につき1回だけ流す（4案ぶんの問い合わせ音声をまとめて取る）
    per_run: list[dict] = []
    for run in runs:
        data = pipe.replay_seats(run, vp, align="text", query=Context())
        if data is None:
            continue
        got = {name: outcomes(data, q, pick) for name, q, pick in VARIANTS}
        per_run.append(got)
        print(f"# {run} 済み（{len(data['steps'])}発話）", flush=True)
    if not per_run:
        raise SystemExit("# 測れるランが無い")

    def _pool(subset):
        return {name: [r for g in subset for r in g[name]]
                for name, _q, _p in VARIANTS}

    all_rows = _pool(per_run)
    _table(f"全体（{len(per_run)}本）", all_rows)
    _table("1秒未満だけ", all_rows, keep=lambda r: r["dur_ms"] < 1000)
    _table("1秒以上だけ（壊していないかの確認）", all_rows,
           keep=lambda r: r["dur_ms"] >= 1000)
    if 0 < args.split < len(per_run):
        _table(f"開発（{args.split}本）", _pool(per_run[:args.split]))
        _table(f"検証（{len(per_run) - args.split}本）",
               _pool(per_run[args.split:]))

    print("\n読み方: 誤帰属が減っても未確定が同じだけ増えるなら、それは")
    print("  「間違えるより黙る」への置き換えであって、当たるようになった")
    print("  わけではない。正解の列が上がって初めて改善である。")


if __name__ == "__main__":
    main()
