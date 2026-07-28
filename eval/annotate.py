#!/usr/bin/env python3
"""話者の正解を付けるための注釈ツール（ローカルサーバ＋ブラウザ画面）.

    uv run python eval/annotate.py 2026-07-25_1723        # 収録セッション
    uv run python eval/annotate.py ~/audio/kaigi.mp3      # 任意の音声
    uv run python eval/annotate.py <対象> --minutes 10    # 頭の10分だけ

ブラウザが開いたら、<kbd>1</kbd>〜<kbd>9</kbd> で話者、<kbd>S</kbd> で直前と
同じ話者、<kbd>0</kbd> 不明、<kbd>M</kbd> 複数人。既定の「自動送り」では
1区間鳴らして止まり、キーを押すと次へ進んで自動で鳴る——耳と指だけで進む。

**旧 `make_gt_annotator.py` との違い**

- 音声をサーバから配る。ファイル選択もパス合わせも要らない。
- 保存が自動。押すたびに `eval/gt_<名前>.json` へ書き、開き直せば続きから。
  （ダウンロードして自分で置き直す手間と、消えてしまう事故が無くなる）
- 話者は3人固定でなく9人まで。4人以上の会議や雑談にそのまま使える。
- 収録セッション以外の音声も扱える。文字起こしが無い音声は無音で区切る。
- 全体波形を出し、ラベル済みの区間を色で示す。どこが残っているか一目で分かる。

出力は従来と同じ `{"labels": {区間ID: "S1"|...|"MULTI"|"UNK"}}` 形式なので、
`eval/decompose_attribution.py` などの採点はそのまま動く。収録セッションでは
区間IDに turns の `turn_id` を使う（従来のGTと互換）。

注意: 2026-07-25 以前に収録した音声は、録音wavと発話msがずれている場合が
ある（送信できなかった音声まで録音に書いていたため。修正済み）。ずれた録音に
注釈を付けても採点には使えないので、実ドメインの測定には修正後の録音を使う。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _annot_html import PAGE  # noqa: E402
from _gtlib import read_jsonl  # noqa: E402

SR = 16000
CACHE = ROOT / "eval" / "_annot_audio"


# ----------------------------------------------------------------------
# 音声の用意
# ----------------------------------------------------------------------
def prepare_audio(src: Path, minutes: float | None) -> tuple[Path, np.ndarray]:
    """16kHz モノラル wav を用意して (パス, 波形) を返す.

    そのまま条件を満たす wav は触らない（元の録音を書き換えない）。それ以外は
    ffmpeg で変換して `eval/_annot_audio/` に置く。ffmpeg が無い環境では、
    mp3/m4a などを開けない旨だけ伝えて終わる——黙って壊れた音を配るよりよい。
    """
    need_convert = True
    if src.suffix.lower() == ".wav":
        try:
            with wave.open(str(src)) as w:
                need_convert = not (w.getframerate() == SR
                                    and w.getnchannels() == 1
                                    and w.getsampwidth() == 2)
        except wave.Error:
            need_convert = True
    if need_convert:
        if shutil.which("ffmpeg") is None:
            sys.exit(f"# {src.name} を 16kHz モノラルwav に変換できません"
                     f"（ffmpeg が要ります）")
        CACHE.mkdir(parents=True, exist_ok=True)
        dst = CACHE / (re.sub(r"[^\w.-]", "_", src.stem) + ".wav")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
               "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(dst)]
        subprocess.run(cmd, check=True)
        src = dst

    with wave.open(str(src)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    y = pcm.astype(np.float32) / 32768.0

    if minutes is not None and len(y) > int(minutes * 60 * SR):
        CACHE.mkdir(parents=True, exist_ok=True)
        y = y[:int(minutes * 60 * SR)]
        cut = CACHE / (src.stem + f"_first{minutes:g}min.wav")
        with wave.open(str(cut), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((y * 32767).astype("<i2").tobytes())
        src = cut
    return src, y


# ----------------------------------------------------------------------
# 区間の用意
# ----------------------------------------------------------------------
def segments_from_turns(path: Path, limit_sec: float) -> list[dict]:
    """収録セッションの turns から区間を作る（IDは turn_id で従来GTと互換）."""
    out = []
    for t in read_jsonl(path):
        if t["ms"] / 1000 >= limit_sec:
            break
        out.append({"id": str(t["turn_id"]), "start": t["ms"] / 1000,
                    "end": min(t["end_ms"] / 1000, limit_sec),
                    "text": t.get("text", ""), "sys": t.get("speaker", "")})
    return out


def segments_from_vad(y: np.ndarray, *, frame: float = 0.02,
                      min_sec: float = 0.4, gap_sec: float = 0.35,
                      max_sec: float = 12.0) -> list[dict]:
    """文字起こしが無い音声を無音で区切る.

    しきい値は録音ごとに音量が違うので、フレームRMSの分布から決める
    （中央値と上位1割の間）。短すぎる区間は落とし、近い区間はつなぎ、
    長すぎる区間は一番静かな所で割る——注釈する側が1区間＝1話者として
    扱えるようにするため。
    """
    f = int(frame * SR)
    n = len(y) // f
    if n == 0:
        return []
    rms = np.sqrt((y[:n * f].reshape(n, f) ** 2).mean(1))
    lo, hi = np.percentile(rms, 50), np.percentile(rms, 90)
    thr = max(lo + (hi - lo) * 0.35, 1e-4)
    on = rms > thr

    spans: list[list[int]] = []
    i = 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        spans.append([i, j + 1])
        i = j + 1
    merged: list[list[int]] = []
    for s in spans:
        if merged and (s[0] - merged[-1][1]) * frame <= gap_sec:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    out: list[dict] = []
    for a, b in merged:
        if (b - a) * frame < min_sec:
            continue
        pieces = [(a, b)]
        while True:
            long = [p for p in pieces if (p[1] - p[0]) * frame > max_sec]
            if not long:
                break
            for p in long:
                pieces.remove(p)
                mid_lo = p[0] + int(max_sec * 0.4 / frame)
                mid_hi = min(p[1] - int(min_sec / frame), p[0] + int(max_sec / frame))
                if mid_hi <= mid_lo:
                    pieces.append((p[0], p[1]))
                    break
                cut = mid_lo + int(np.argmin(rms[mid_lo:mid_hi]))
                pieces += [(p[0], cut), (cut, p[1])]
            else:
                continue
            break
        for x, z in sorted(pieces):
            out.append({"start": round(x * frame, 2), "end": round(z * frame, 2),
                        "text": "", "sys": ""})
    for k, s in enumerate(out, 1):
        s["id"] = str(k)
    return out


def peaks_of(y: np.ndarray, n: int = 1400) -> list[float]:
    """波形表示用に、区間ごとの最大振幅へ間引く."""
    if len(y) == 0:
        return []
    step = max(1, len(y) // n)
    m = len(y) // step
    a = np.abs(y[:m * step].reshape(m, step)).max(1)
    top = float(a.max()) or 1.0
    return [round(float(v / top), 3) for v in a]


# ----------------------------------------------------------------------
# サーバ
# ----------------------------------------------------------------------
class _State:
    def __init__(self, name, title, wav, segments, peaks, duration, gt_path,
                 seg_path):
        self.name = name
        self.title = title
        self.wav = wav
        self.segments = segments
        self.peaks = peaks
        self.duration = duration
        self.gt_path = gt_path
        self.seg_path = seg_path
        self.lock = threading.Lock()
        self.backup_done = False

    def load_gt(self) -> dict:
        if self.gt_path.exists():
            try:
                return json.loads(self.gt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {}

    def save_gt(self, names: dict, labels: dict) -> None:
        """GTを書き戻す。既存の項目は残し、初回だけ `.bak` を取る.

        自動保存なので、間違って開いただけで前の正解が消えると取り返しが
        つかない。既存の doc に上書きする形にして知らない項目
        （`transplanted_from` など）を守り、書き込みは一時ファイル経由で
        差し替える（途中で落ちても壊れた JSON が残らない）。
        """
        import datetime
        with self.lock:
            doc = self.load_gt()
            if doc and not self.backup_done:
                bak = self.gt_path.with_suffix(".json.bak")
                if not bak.exists():
                    bak.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
                self.backup_done = True
            doc.setdefault("session", self.name)
            doc["created"] = datetime.datetime.now().isoformat(timespec="seconds")
            doc["speaker_names"] = names
            doc["labels"] = {k: v for k, v in labels.items() if v}
            doc["labeled"] = len(doc["labels"])
            doc["total"] = max(len(self.segments), doc.get("total", 0))
            tmp = self.gt_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(self.gt_path)


class _Handler(BaseHTTPRequestHandler):
    state: _State

    def log_message(self, *a):  # アクセスログは出さない
        pass

    def _send(self, code, body: bytes, ctype: str, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        s = self.state
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/info"):
            gt = s.load_gt()
            names = gt.get("speaker_names") or {
                f"S{i}": f"話者{i}" for i in (1, 2, 3)}
            body = {"title": s.title, "segments": s.segments, "peaks": s.peaks,
                    "duration": s.duration, "labels": gt.get("labels") or {},
                    "speaker_names": names,
                    "gt_path": str(s.gt_path.relative_to(ROOT))}
            self._send(200, json.dumps(body, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/audio"):
            self._send_audio()
        elif self.path.startswith("/labels"):
            body = json.dumps(s.load_gt(), ensure_ascii=False, indent=1).encode()
            self._send(200, body, "application/json; charset=utf-8",
                       [("Content-Disposition",
                         f'attachment; filename="{s.gt_path.name}"')])
        else:
            self._send(404, b"not found", "text/plain")

    def _send_audio(self):
        """Range 付きで wav を返す（付けないとブラウザが頭出しできない）."""
        data = self.state.wav.read_bytes()
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        if not m:
            self._send(200, data, "audio/wav", [("Accept-Ranges", "bytes")])
            return
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else len(data) - 1
        b = min(b, len(data) - 1)
        chunk = data[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def do_POST(self):
        if not self.path.startswith("/labels"):
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            doc = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            self._send(400, b'{"error":"bad json"}', "application/json")
            return
        self.state.save_gt(doc.get("speaker_names") or {},
                           doc.get("labels") or {})
        out = {"ok": True,
               "path": str(self.state.gt_path.relative_to(ROOT))}
        self._send(200, json.dumps(out, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")


# ----------------------------------------------------------------------
def build_state(target: str, minutes: float | None) -> _State:
    session_wav = ROOT / "transcripts" / f"{target}.wav"
    turns = ROOT / "transcripts" / f"{target}.turns.jsonl"
    if session_wav.exists():
        name, src = target, session_wav
    else:
        src = Path(target).expanduser()
        if not src.exists():
            sys.exit(f"# {target} が見つかりません"
                     f"（収録セッション名か音声ファイルのパスを指定してください）")
        name = re.sub(r"[^\w.-]", "_", src.stem)

    wav, y = prepare_audio(src, minutes)
    duration = len(y) / SR
    if session_wav.exists() and turns.exists():
        segs = segments_from_turns(turns, duration)
        title = f"{target}（収録セッション）"
        seg_path = None
    else:
        segs = segments_from_vad(y)
        title = f"{src.name}（無音で自動区切り）"
        seg_path = ROOT / "eval" / f"segments_{name}.json"
        seg_path.write_text(json.dumps(segs, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    if not segs:
        sys.exit("# 区切れる区間がありませんでした（音声が短い/無音の可能性）")
    return _State(name, title, wav, segs, peaks_of(y), duration,
                  ROOT / "eval" / f"gt_{name}.json", seg_path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("target", help="収録セッション名、または音声ファイルのパス")
    p.add_argument("--minutes", type=float, default=None,
                   help="頭から何分だけを対象にするか（既定: 全部）")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="ブラウザを開かない")
    args = p.parse_args(argv)

    state = build_state(args.target, args.minutes)
    _Handler.state = state
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"# {state.title}")
    print(f"#   区間 {len(state.segments)} 件・{state.duration / 60:.1f}分")
    if state.seg_path is not None:
        print(f"#   区切り: {state.seg_path.relative_to(ROOT)}")
    print(f"#   保存先: {state.gt_path.relative_to(ROOT)}（自動保存）")
    print(f"# {url} を開いてください（Ctrl-C で終了）")
    if not args.no_open:
        with __import__("contextlib").suppress(Exception):
            webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n# 終了しました")


if __name__ == "__main__":
    main()
