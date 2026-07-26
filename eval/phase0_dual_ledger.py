#!/usr/bin/env python3
"""Phase 0: 二重帳簿根治の設計案A/Bの反実仮想採点ハーネス.

docs/design/handoff_2026-07-25_dual_ledger_rootcure.md §3 Phase 0 の測定専用
スクリプト。**本体（src/）の挙動は一切変更しない**。

測定の3層:
  1. リンク判定（案Bの核）: 記録済みランの各鋳造（自動登録）イベントについて、
     「新人物のプロファイル vs 席持ちクラスタ(@diar:N)の蓄積声紋」の対称類似を
     wav から再計算し、同一人物か（GT/疑似GT）と突き合わせて閾値の精度曲線を出す。
  2. 発話レベル案B: リンク成立の鋳造を「新席を作らず統合」に差し替えた鍵系列を
     SessionState 実物の統一席ルールに流し、実質精度/誤帰属/未確定を再採点。
  3. 発話レベル案A: classify 再生で STT側鋳造を停止し、diag のクラスタ系列＋
     wav 再計算でクラスタ側鋳造（クリーン累積>=NAMING_MIN_SEC・既存の誰とも
     dedupe未満で鋳造→即confirm）をエミュレートして同様に再採点。

既知の近似（数字の読みで必ず考慮すること）:
  - クラスタの蓄積音声は diag で key=@diar:N と観測できた発話に限る
    （声紋勝ち発話のクラスタ所属は記録されない→蓄積は過小＝保守側）。
  - 案Aは登録動態そのものが変わるため、記録済み入力への反応のみの近似
    （系のフィードバックは再現できない）。
  - 実会話5本は GT が無いため、全量音声から作った人物別声紋を疑似GTにする。

使い方（クラウド検証環境 or Mac、リポジトリ直下で）:
    uv run python eval/phase0_dual_ledger.py --stage links
    uv run python eval/phase0_dual_ledger.py --stage links --sessions 2026-07-25_1723
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402  (eval/ 共通採点)

from das.asr.live._constants import (  # noqa: E402
    _BACKCHANNEL_RE,
    PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC,
    PYANNOTE_CLUSTER_NAMING_MIN_SEC,
    SR,
    UNSURE_SPEAKER,
)

# 信頼4種（_attribution._VOICEPRINT_RELIABLE_KINDS と同一。import しないのは
# アンダースコア私有名への依存を測定スクリプトに持ち込まないため）
RELIABLE_KINDS = {"声紋一致", "補正", "自動登録", "合流"}

# GT付きハイブリッド11本（§18.10 の 14本から非ハイブリッドの CallHome 3本を除外）
GT_SESSIONS = [
    "2026-07-14_142016", "2026-07-16_1723",
    "2026-07-20_1623", "2026-07-20_1635", "2026-07-20_1655",
    "2026-07-20_1709", "2026-07-20_1723", "2026-07-20_1738",
    "2026-07-20_1748", "2026-07-20_2341", "2026-07-20_2351",
]
# GT無し実会話5本（ハイブリッド）
REAL_SESSIONS = [
    "2026-07-22_223337", "2026-07-25_1534", "2026-07-25_1545",
    "2026-07-25_1641", "2026-07-25_1723",
]

# 反実仮想で使う想定話者数（＝実際の参加人数）。
# ライブランの設定値をそのまま使わない理由: 07-25 の実会話3本（1534/1545/1641）は
# 「上限1のまま2人会話」の実事故ラン（handoff §20-3, 修正 76b071b）で、当時の
# constrain も統一席ルール以前の旧実装。設計案の比較は「正しく設定された条件」で
# 行う必要があるため、実際の参加人数を与える。Chiba系はすべて3人会話。
MAX_SPEAKERS = {s: 3 for s in GT_SESSIONS} | {s: 2 for s in REAL_SESSIONS}


# ------------------------------------------------------------------
# 入力
# ------------------------------------------------------------------

def read_audio(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == SR
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


class Session:
    """1セッション分の記録（diag系列・turns・GT・音声）."""

    def __init__(self, name: str):
        self.name = name
        root = ROOT / "transcripts"
        self.utts: list[dict] = []          # diag の発話行（時系列）
        self.cluster_events: list[dict] = []
        self.max_speakers: int | None = None
        with open(root / f"{name}.diag.jsonl", encoding="utf-8") as f:
            diag_lines = f.readlines()
        for line in diag_lines:
            d = json.loads(line)
            t = d.get("type")
            if t == "cluster_naming":
                self.cluster_events.append(d)
            elif t == "constrain_drop":
                self.max_speakers = d.get("max_speakers")
            elif t is None and "label" in d and "key" in d:
                self.utts.append(d)
                if d.get("kind") == "話者数上限":
                    self.max_speakers = d.get("max_speakers", self.max_speakers)
        self.text_by_ms: dict[int, str] = {}
        for t in _gtlib.read_jsonl(root / f"{name}.turns.jsonl"):
            self.text_by_ms.setdefault(t["ms"], t.get("text", ""))
        self.audio = read_audio(root / f"{name}.wav")
        # GT（あれば）: turn_id→S1.. を発話開始msに引き直す
        self.gt_by_ms: dict[int, str] = {}
        gt_path = ROOT / "eval" / f"gt_{name}.json"
        if gt_path.exists():
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            turns = _gtlib.read_jsonl(root / f"{name}.turns.jsonl")
            for t in turns:
                code = gt["labels"].get(str(t["turn_id"]))
                if code is not None:
                    self.gt_by_ms[t["ms"]] = code

    def wav_of(self, d: dict) -> np.ndarray:
        s = int(d["ms"] * SR / 1000)
        e = int(d["end"] * SR / 1000)
        return self.audio[s:e]

    def text_of(self, d: dict) -> str:
        return self.text_by_ms.get(d["ms"], "")

    def is_bc(self, d: dict) -> bool:
        return bool(_BACKCHANNEL_RE.match(self.text_of(d).strip()))


# ------------------------------------------------------------------
# 埋め込み（ReDimNet, セッション内キャッシュ）
# ------------------------------------------------------------------

class Embedder:
    def __init__(self, model: str = "redimnet"):
        import os
        import tempfile

        from das.asr.live._voice_profiles import VoiceProfiles
        path = os.path.join(tempfile.mkdtemp(prefix="phase0_vp_"), "voices.json")
        self.vp = VoiceProfiles(path=path, model=model)
        self._cache: dict[tuple, np.ndarray | None] = {}

    def embed(self, wav: np.ndarray) -> np.ndarray | None:
        if wav.size < int(SR * 0.4):
            return None
        key = (wav.size, hash(wav.tobytes()))
        if key not in self._cache:
            self._cache[key] = self.vp._embed(wav)
        return self._cache[key]


def concat_tail(segs: list[np.ndarray], cap_sec: float) -> np.ndarray:
    """観測順の音声リストを、末尾から cap_sec 秒に収めて連結する（クラスタbuffer相当）."""
    out: list[np.ndarray] = []
    total = 0
    for s in reversed(segs):
        out.append(s)
        total += s.size
        if total >= cap_sec * SR:
            break
    return np.concatenate(list(reversed(out))) if out else np.empty(0, np.float32)


# ------------------------------------------------------------------
# ステージ1: リンク判定（鋳造イベント × 席クラスタ）
# ------------------------------------------------------------------

def person_certified_audio(sess: Session, person: str, *, after_ms: int | None = None,
                           before_ms: int | None = None, limit_sec: float = 30.0
                           ) -> list[np.ndarray]:
    """ライブが信頼4種で人物Pと裏付けた発話音声（時系列）を集める.

    kind が信頼4種かつ照合先 name==P（自動登録は key==P）の発話は、
    ライブの声紋層が「Pの声」と証明した音声＝Pの声の確かな標本。
    """
    segs, total = [], 0.0
    for d in sess.utts:
        if after_ms is not None and d["ms"] < after_ms:
            continue
        if before_ms is not None and d["ms"] >= before_ms:
            continue
        if d.get("kind") in RELIABLE_KINDS and (
                d.get("name") == person or d.get("key") == person):
            w = sess.wav_of(d)
            if w.size >= SR * 1.0:
                segs.append(w)
                total += w.size / SR
                if total >= limit_sec:
                    break
    return segs


def seat_audio(sess: Session, seat: str, *, before_ms: int | None = None,
               ) -> list[np.ndarray]:
    """席クラスタ @diar:N の観測音声（diag key==seat の発話, 時系列）."""
    return [sess.wav_of(d) for d in sess.utts
            if d.get("key") == seat and (before_ms is None or d["ms"] < before_ms)
            and sess.wav_of(d).size >= SR * 0.5]


def gt_majority(sess: Session, condition) -> tuple[str | None, int, float]:
    """条件に合う発話のGT多数派コードと (票数, 純度) を返す（GT付きセッション用）.

    票数が少ない・純度が低い多数決は同一人物判定の根拠にならない（Chibaでは
    pyannoteイベントとSoniox区切りの整列ずれで席発話のGTが混ざる, §18.6）。
    呼び出し側は票数>=3・純度>=0.7 を確信条件として使う。
    """
    votes = Counter()
    for d in sess.utts:
        if condition(d):
            code = sess.gt_by_ms.get(d["ms"])
            if code in ("S1", "S2", "S3"):
                votes[code] += 1
    if not votes:
        return None, 0, 0.0
    code, top = votes.most_common(1)[0]
    total = sum(votes.values())
    return code, total, top / total


def session_calibration(sess: Session, emb: Embedder) -> dict:
    """セッション内の同一/別人の類似分布を実測する（閾値の移植可能性の検査）.

    - 同一人物: 各人物Pの信頼4種音声を前半/後半に割った split-half 類似
    - 別人: GTコードが異なる人物ペアの全量プロファイル類似
    チャネル（マイク直 / スピーカー再生 / 電話品質）で絶対値が動くため、
    リンク閾値が全セッション共通で成立するかはこの分布で判断する。
    """
    persons: dict[str, list[np.ndarray]] = {}
    for d in sess.utts:
        if d.get("kind") in RELIABLE_KINDS:
            p = d.get("name") or d.get("key")
            if isinstance(p, str) and p.startswith("人物"):
                w = sess.wav_of(d)
                if w.size >= SR * 1.0:
                    persons.setdefault(p, []).append(w)
    same, diff = [], []
    profs: dict[str, np.ndarray] = {}
    codes: dict[str, tuple[str | None, int, float]] = {}
    for p, segs in persons.items():
        # 音声長は必ず打ち切る（無制限に連結すると10分級の波形になり、
        # 埋め込み1回に分単位かかる＝測定が終わらない）。
        if len(segs) >= 4:
            half = len(segs) // 2
            e1 = emb.embed(concat_tail(segs[:half], 30.0))
            e2 = emb.embed(concat_tail(segs[half:], 30.0))
            if e1 is not None and e2 is not None:
                same.append(float(np.dot(e1, e2)))
        e = emb.embed(concat_tail(segs, 60.0))
        if e is not None:
            profs[p] = e
        codes[p] = gt_majority(
            sess, lambda x, p=p: x.get("kind") in RELIABLE_KINDS
            and (x.get("name") == p or x.get("key") == p))
    names = sorted(profs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, na, pa = codes[a]
            cb, nb, pb = codes[b]
            if (ca and cb and ca != cb and na >= 3 and nb >= 3
                    and pa >= 0.7 and pb >= 0.7):
                diff.append(float(np.dot(profs[a], profs[b])))
    return {"same": [round(x, 3) for x in same],
            "diff": [round(x, 3) for x in diff]}


def stage_linkops(links_path: Path, emb: Embedder) -> None:
    """実装どおりの演算で対称類似を測り直す（閾値の移植可能性の検査）.

    ステージ1の sim_full は「音声を連結してから声紋化」で測ったが、実装で
    人物側に使える蓄積は ``VoiceProfiles.own_embs``（受理一致の**発話ごとの
    声紋**を最大16件）である。演算が違えば同じ数字は出ない可能性があるため、
    同じペアを実装どおりの式で測り直す:

        cluster側 = 声紋(連結したクラスタbuffer)   ← match_profile と同じ
        person側  = 正規化(mean(発話ごとの声紋))   ← own_embs の平均と同じ

    出力は元の JSONL に ``sim_op`` を足して書き戻す。
    """
    rows = [json.loads(x) for x in links_path.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    by_session: dict[str, list[dict]] = {}
    for r in rows:
        if not r.get("empty"):
            by_session.setdefault(r["session"], []).append(r)
    for name, rs in by_session.items():
        sess = Session(name)
        person_vec: dict[str, np.ndarray | None] = {}
        seat_vec: dict[str, np.ndarray | None] = {}
        for r in rs:
            p = r["person"]
            if p not in person_vec:
                # own_embs 相当: 受理一致の発話ごと声紋を最大 _OWN_EMB_CAP(16) 件
                embs = [emb.embed(w) for w in
                        person_certified_audio(sess, p, limit_sec=10 ** 6)[-16:]]
                embs = [e for e in embs if e is not None]
                if embs:
                    v = np.mean(np.stack(embs), axis=0)
                    person_vec[p] = v / np.linalg.norm(v)
                else:
                    person_vec[p] = None
            s = r["seat"]
            if s not in seat_vec:
                w = concat_tail(seat_audio(sess, s),
                                PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC)
                seat_vec[s] = emb.embed(w) if w.size else None
            pv, sv = person_vec[p], seat_vec[s]
            r["sim_op"] = (round(float(np.dot(pv, sv)), 3)
                           if pv is not None and sv is not None else None)
        print(f"  {name}: {len(rs)}ペア 測り直し", flush=True)
    links_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")


def stage_linksim(sessions: list[str], emb: Embedder, out_path: Path) -> None:
    """案B遅延を時系列で忠実にシミュレートし、判定点ごとの正誤を出す.

    ステージ1/linkops は「セッション全量の声紋」で測ったが、実装が実際に
    比較できるのは**その時点のバッファ**である（クラスタ側は直近
    PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC 秒。1723 で全量なら 0.73 の
    ペアが直近20秒だと 0.05 になる＝クラスタ後半が別人の声、という例がある）。
    さらに遅延リンクは「蓄積が伸びるたびに何度も判定する」ため、1回でも
    誤って跨げば統合が起きる。よって測るべきは2つ:

      - 初回成立の正誤（実際に統合が起きる瞬間）
      - 全判定点のうち誤って跨いだ回数（繰り返し判定による誤リンク露出）

    判定点は各クラスタの累積音声が 5/10/20/40/80/160 秒を超えた発話
    （埋め込み計算量を抑えるための間引き。実装は毎観測で判定するが、
    バッファは直近20秒なので情報の増分はこの粒度でほぼ尽きる）。
    """
    checkpoints = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
    rows: list[dict] = []
    for name in sessions:
        sess = Session(name)
        truth = {}
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("session") == name and not r.get("empty"):
                    truth[(r["person"], r["seat"])] = r["same_person"]
        seat_buf: dict[str, list[np.ndarray]] = {}
        seat_sec: dict[str, float] = {}
        seat_next: dict[str, int] = {}
        person_buf: dict[str, list[np.ndarray]] = {}
        person_sec: dict[str, float] = {}
        n_pts = 0
        for d in sess.utts:
            k = str(d.get("key", ""))
            w = sess.wav_of(d)
            if d.get("kind") in RELIABLE_KINDS:
                p = str(d.get("name") or d.get("key"))
                if p.startswith("人物") and w.size >= SR * 1.0:
                    person_buf.setdefault(p, []).append(w)
                    person_sec[p] = person_sec.get(p, 0.0) + w.size / SR
            if not k.startswith("@diar:") or w.size < SR * 0.5:
                continue
            seat_buf.setdefault(k, []).append(w)
            seat_sec[k] = seat_sec.get(k, 0.0) + w.size / SR
            i = seat_next.get(k, 0)
            if i >= len(checkpoints) or seat_sec[k] < checkpoints[i]:
                continue
            seat_next[k] = i + 1
            cv = emb.embed(concat_tail(seat_buf[k],
                                       PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC))
            if cv is None:
                continue
            for p, sec in person_sec.items():
                if sec < PYANNOTE_CLUSTER_NAMING_MIN_SEC:
                    continue
                pv = emb.embed(concat_tail(
                    person_buf[p], PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC))
                if pv is None:
                    continue
                n_pts += 1
                rows.append({
                    "session": name, "person": p, "seat": k,
                    "at_ms": d["ms"], "seat_sec": round(seat_sec[k], 1),
                    "person_sec": round(sec, 1),
                    "sim": round(float(np.dot(pv, cv)), 3),
                    "same_person": truth.get((p, k)),
                })
        print(f"  {name}: 判定点{n_pts}件", flush=True)
    (out_path.parent / "phase0_linksim.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")


def stage_links(sessions: list[str], emb: Embedder, out_path: Path) -> None:
    """全鋳造イベントについて、席クラスタとの鋳造時対称類似と同一人物判定を出す.

    セッションごとに out_path へ追記する（済みセッションはスキップ＝再開可能）。
    """
    done: set[str] = set()
    rows: list[dict] = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add(r["session"])
    for name in sessions:
        if name in done:
            print(f"  {name}: 済み（スキップ）", flush=True)
            continue
        sess = Session(name)
        has_gt = bool(sess.gt_by_ms)
        seats_seen: list[str] = []
        mints = []
        for d in sess.utts:
            k = d.get("key", "")
            if isinstance(k, str) and k.startswith("@diar:") and k not in seats_seen:
                seats_seen.append(k)
            if d.get("kind") == "自動登録":
                mints.append((d, list(seats_seen)))
        new_rows: list[dict] = []
        for d, seats in mints:
            person = d.get("name") or d.get("key")
            t_mint = d["ms"]
            # 鋳造時点の人物プロファイル近似: 鋳造以降にライブが信頼4種で
            # Pと裏付けた最初の~10秒（鋳造直後のPの声の確かな標本）
            p_segs = person_certified_audio(sess, person, after_ms=t_mint,
                                            limit_sec=10.0)
            p_emb = emb.embed(np.concatenate(p_segs)) if p_segs else None
            # 事後検証用のP全量プロファイル（セッション全体、上限30秒）
            p_full = person_certified_audio(sess, person, limit_sec=30.0)
            p_full_emb = emb.embed(np.concatenate(p_full)) if p_full else None
            for seat in seats:
                # 鋳造時点の席クラスタ蓄積（buffer上限20秒の意味論に合わせ末尾20秒）
                s_segs = seat_audio(sess, seat, before_ms=t_mint)
                s_wav = concat_tail(s_segs, PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC)
                s_emb = emb.embed(s_wav) if s_wav.size else None
                # 事後検証用の席全量
                s_all = seat_audio(sess, seat)
                s_full = concat_tail(s_all, 60.0)
                s_full_emb = emb.embed(s_full) if s_full.size else None
                sim_mint = (float(np.dot(p_emb, s_emb))
                            if p_emb is not None and s_emb is not None else None)
                sim_full = (float(np.dot(p_full_emb, s_full_emb))
                            if p_full_emb is not None and s_full_emb is not None
                            else None)
                # 同一人物か: GTセッションは多数派GTコードの一致、GT無しは全量類似
                same = None
                basis = None
                if has_gt:
                    g_p, n_p, pu_p = gt_majority(
                        sess, lambda x, p=person: x.get("kind") in RELIABLE_KINDS
                        and (x.get("name") == p or x.get("key") == p))
                    g_s, n_s, pu_s = gt_majority(
                        sess, lambda x, s=seat: x.get("key") == s)
                    if (g_p and g_s and n_p >= 3 and n_s >= 3
                            and pu_p >= 0.7 and pu_s >= 0.7):
                        same = (g_p == g_s)
                        basis = (f"gt:{g_p}({n_p},{pu_p:.0%})"
                                 f"vs{g_s}({n_s},{pu_s:.0%})")
                    elif g_p and g_s:
                        basis = (f"gt弱:{g_p}({n_p},{pu_p:.0%})"
                                 f"vs{g_s}({n_s},{pu_s:.0%})")
                if same is None and sim_full is not None:
                    # 実測分離: 同一0.61-0.70 / 別人<=0.31（handoff §1）。中間帯は判定保留
                    if sim_full >= 0.50:
                        same = True
                    elif sim_full <= 0.38:
                        same = False
                    basis = f"fullsim:{sim_full:.3f}"
                new_rows.append({
                    "session": name, "person": person, "mint_ms": t_mint,
                    "seat": seat,
                    "sim_mint": None if sim_mint is None else round(sim_mint, 3),
                    "sim_full": None if sim_full is None else round(sim_full, 3),
                    "p_sec_mint": round(sum(x.size for x in p_segs) / SR, 1),
                    "s_sec_mint": round(s_wav.size / SR, 1) if s_wav.size else 0.0,
                    "same_person": same, "basis": basis,
                })
        cal = session_calibration(sess, emb)
        cal_note = (f" 分離[同一{min(cal['same']):.2f}-{max(cal['same']):.2f}"
                    if cal["same"] else " 分離[同一n/a")
        cal_note += (f" / 別人{min(cal['diff']):.2f}-{max(cal['diff']):.2f}]"
                     if cal["diff"] else " / 別人n/a]")
        for r in new_rows:
            r["cal"] = cal
        rows.extend(new_rows)
        with open(out_path, "a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not new_rows:   # 鋳造ゼロでも済み印を残す
                f.write(json.dumps({"session": name, "empty": True},
                                   ensure_ascii=False) + "\n")
        print(f"  {name}: 鋳造{len(mints)}件 × 席 → {len(new_rows)}ペア{cal_note}",
              flush=True)
    rows = [r for r in rows if not r.get("empty")]
    # 集計: 閾値sweep
    print(f"\n== リンク判定（鋳造×席 {len(rows)}ペア, 判定可能 "
          f"{sum(1 for r in rows if r['same_person'] is not None)}件） ==")
    print(f"{'th':>5} {'リンク':>4} {'正':>3} {'誤':>3} {'見送り正':>5} {'見送り誤(取り逃し)':>8}")
    for th in (0.35, 0.40, 0.45, 0.50, 0.55):
        link_ok = link_ng = skip_ok = skip_ng = 0
        for r in rows:
            if r["same_person"] is None or r["sim_mint"] is None:
                continue
            linked = r["sim_mint"] >= th
            if linked and r["same_person"]:
                link_ok += 1
            elif linked:
                link_ng += 1
            elif r["same_person"]:
                skip_ng += 1
            else:
                skip_ok += 1
        print(f"{th:>5.2f} {link_ok+link_ng:>4} {link_ok:>3} {link_ng:>3} "
              f"{skip_ok:>5} {skip_ng:>8}")


# ------------------------------------------------------------------
# ステージ2/3: 発話レベルの反実仮想採点（現状 / 案B / 案A-lite）
# ------------------------------------------------------------------

def make_state(max_speakers: int | None, tmp: Path):
    """統一席ルール replay 用の SessionState 実物を作る."""
    import datetime
    from types import SimpleNamespace

    from das.asr.live._session_state import SessionState
    tmp.mkdir(parents=True, exist_ok=True)
    return SessionState(
        args=SimpleNamespace(diarization_max_speakers=max_speakers),
        started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp / "o.md"), html_path=str(tmp / "o.html"),
        diag_path=str(tmp / "o.diag"), turns_path=str(tmp / "o.turns"),
        wav_path=str(tmp / "o.wav"))


def replay_seats(sess: Session, keys: list[str],
                 renames: dict[int, list[tuple[str, str]]],
                 max_speakers: int | None, tmp: Path,
                 *, disp: bool = False) -> list[str]:
    """鍵系列を SessionState 実物の統一席ルールに流し、最終話者列を返す.

    flush の順序を模す: rekey（鋳造リネーム等）→ constrain → records追記 →
    disp_name（席文字の割り当て）。records は rekey で遡及的に書き替わるため、
    返すのは全処理後の records の speaker 列（＝ライブの最終状態と同じ意味論）。
    ``disp=True`` なら表示名（参加者A/未確定…）に変換して返す（turns.jsonl と
    突き合わせて再現の忠実性を検査するため）。
    """
    s = make_state(max_speakers, tmp)
    for i, d in enumerate(sess.utts):
        for old, new in renames.get(i, []):
            if old != new:
                s.rekey(old, new)
        k = UNSURE_SPEAKER if sess.is_bc(d) else keys[i]
        final = s.constrain_human_speaker_key(k)
        s.records.append({"ms": d["ms"], "end_ms": d["end"],
                          "speaker": final, "text": sess.text_of(d)})
        s.disp_name(final)
    out = [str(r["speaker"]) for r in s.records if "speaker" in r]
    return [s.disp_name(k) for k in out] if disp else out


class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        while self.p.get(x, x) != x:
            x = self.p[x]
        return x

    def union(self, child: str, parent: str) -> None:
        self.p[self.find(child)] = self.find(parent)


def load_links(links_path: Path) -> dict[tuple[str, str], list[dict]]:
    """stage links の出力を (session, person)→[ペア行] で引けるようにする."""
    out: dict[tuple[str, str], list[dict]] = {}
    for line in links_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("empty"):
            continue
        out.setdefault((r["session"], r["person"]), []).append(r)
    return out


def build_raw_seat_map(sess: Session) -> dict[str, str]:
    """cluster_naming イベントと同msの発話行から raw cluster → 席キーの対応を引く."""
    key_by_ms = {d["ms"]: d.get("key") for d in sess.utts}
    m: dict[str, str] = {}
    for ev in sess.cluster_events:
        k = key_by_ms.get(ev.get("ms"))
        if isinstance(k, str) and k.startswith("@diar:"):
            m.setdefault(ev["cluster"], k)
    return m


def confirm_events(sess: Session) -> list[tuple[int, str, str]]:
    """クラスタ確定 (発話index, raw_cluster, 確定名) を diag から検出する."""
    out = []
    ms_to_idx = {d["ms"]: i for i, d in enumerate(sess.utts)}
    for ev in sess.cluster_events:
        m = ev.get("match")
        if (isinstance(m, (list, tuple)) and len(m) == 2
                and m[1] is not None and float(m[1]) >= 0.65):
            idx = ms_to_idx.get(ev["ms"])
            if idx is not None:
                out.append((idx, ev["cluster"], str(m[0])))
    return out


def scenario_base(sess: Session) -> tuple[list[str], dict[int, list]]:
    """現状の再構成: diag の key 系列＋記録済みリネーム（鋳造/クラスタ確定）."""
    keys = [str(d.get("key")) for d in sess.utts]
    renames: dict[int, list[tuple[str, str]]] = {}
    raw_seat = build_raw_seat_map(sess)
    for i, d in enumerate(sess.utts):
        rn = d.get("rename")
        if d.get("kind") in ("自動登録", "合流") and rn:
            renames.setdefault(i, []).append((str(rn[0]), str(rn[1])))
    for idx, raw, name in confirm_events(sess):
        seat = raw_seat.get(raw)
        if seat is not None and seat != name:
            renames.setdefault(idx, []).append((seat, name))
    return keys, renames


def scenario_b(sess: Session, links: dict, th: float, *,
               field: str = "sim_mint") -> tuple[list[str], dict[int, list]]:
    """案B: リンク成立の鋳造を「新席を作らず席へ統合」に差し替えた系列.

    ``field="sim_mint"``: 鋳造の瞬間に1回だけ対称比較する（案B 即時）。
    ``field="sim_full"``: クラスタ蓄積が育った後の対称比較で判定する
    （案B 遅延＝鋳造後もクラスタが伸びるたびに再照合し、成立したら統合）。
    2版の差が「鋳造時点ではクラスタ音声が足りない」ことの代償を示す。
    """
    uf = UnionFind()
    raw_seat = build_raw_seat_map(sess)
    # 鋳造ごとにリンク判定（鋳造時simが th 以上の最良席）
    link_of: dict[str, str] = {}
    for d in sess.utts:
        if d.get("kind") == "自動登録":
            person = str(d.get("name") or d.get("key"))
            best = None
            for r in links.get((sess.name, person), []):
                if (r.get(field) is not None and r[field] >= th
                        and (best is None or r[field] > best[field])):
                    best = r
            if best is not None:
                link_of[person] = best["seat"]
    keys: list[str] = []
    renames: dict[int, list[tuple[str, str]]] = {}
    for i, d in enumerate(sess.utts):
        kind = d.get("kind")
        rn = d.get("rename")
        if kind == "自動登録":
            person = str(d.get("name") or d.get("key"))
            if person in link_of:
                uf.union(person, link_of[person])
            if rn:
                renames.setdefault(i, []).append(
                    (str(rn[0]), uf.find(str(rn[1]))))
        elif kind == "合流" and rn:
            renames.setdefault(i, []).append((str(rn[0]), uf.find(str(rn[1]))))
        keys.append(uf.find(str(d.get("key"))))
    for idx, raw, name in confirm_events(sess):
        seat = raw_seat.get(raw)
        target = uf.find(name)
        if seat is not None and seat != target:
            renames.setdefault(idx, []).append((seat, target))
    return keys, renames


def scenario_a(sess: Session, links: dict, *, optimistic: bool,
               min_clean_sec: float = 5.0
               ) -> tuple[list[str], dict[int, list], dict]:
    """案A: 鋳造をクラスタ側に一本化した場合の近似系列（悲観/楽観の2版）.

    案Aでは戸籍＝クラスタなので、声紋で人物Pと判定された発話は「Pを鋳造した
    クラスタ」に帰属する。よって記録から作る近似は「人物→席のリンク」で置き換える:
      - 人物→席: 全量声紋の最良リンク（sim_full>=0.50）。リンク不能なら
        その声は案Aでは戸籍を持てない＝未確定。
      - 席同士の合流: 同じ人物にリンクした席は同一人物（dedupe 合流の代理）。
      - 鋳造時刻: 悲観版は「diag で観測できた席発話のクリーン累積が
        min_clean_sec に達した時点」（現行アーキテクチャでは声紋が勝った発話の
        音声はクラスタに蓄積されないため、実際より遅い＝案Aに不利）。
        楽観版は「その席にリンクした人物が実際に鋳造された時点」
        （＝クラスタ側が STT 側と同じ速さで鋳造できた場合）。
    2版の差が、案Aの成績が「クラスタ蓄積をどれだけ速くできるか」に
    どれだけ依存するかを示す。
    """
    seat_utts: dict[str, list[int]] = {}
    for i, d in enumerate(sess.utts):
        k = str(d.get("key"))
        if k.startswith("@diar:"):
            seat_utts.setdefault(k, []).append(i)
    # 悲観版の鋳造時点: 観測できた席音声のクリーン累積
    mint_idx: dict[str, int] = {}
    for seat, idxs in seat_utts.items():
        total = 0.0
        for i in idxs:
            d = sess.utts[i]
            total += (d["end"] - d["ms"]) / 1000.0
            if total >= min_clean_sec:
                mint_idx[seat] = i
                break
    # 人物→席リンク（全量声紋）と、記録上の鋳造 index
    person_seat: dict[str, str] = {}
    for (sname, person), rs in links.items():
        if sname != sess.name:
            continue
        best = None
        for r in rs:
            if (r.get("sim_full") is not None and r["sim_full"] >= 0.50
                    and (best is None or r["sim_full"] > best["sim_full"])):
                best = r
        if best is not None:
            person_seat[person] = best["seat"]
    mint_index_of_person: dict[str, int] = {}
    for i, d in enumerate(sess.utts):
        if d.get("kind") == "自動登録":
            mint_index_of_person.setdefault(
                str(d.get("name") or d.get("key")), i)
    # 席の合流（同じ人物にリンクした席は同一人物）
    uf = UnionFind()
    by_seat: dict[str, list[str]] = {}
    for person, seat in person_seat.items():
        by_seat.setdefault(seat, []).append(person)
    seat_of_person = person_seat
    person_seats: dict[str, list[str]] = {}
    for person, seat in seat_of_person.items():
        person_seats.setdefault(person, []).append(seat)
    for seats in person_seats.values():
        for s2 in seats[1:]:
            uf.union(s2, seats[0])
    if optimistic:
        for person, seat in person_seat.items():
            mi = mint_index_of_person.get(person)
            if mi is not None:
                cur = mint_idx.get(seat)
                mint_idx[seat] = mi if cur is None else min(cur, mi)
    keys: list[str] = []
    stats: Counter = Counter()
    for i, d in enumerate(sess.utts):
        k = str(d.get("key"))
        if k.startswith("@diar:"):
            keys.append(uf.find(k))
            continue
        if k == UNSURE_SPEAKER:
            keys.append(k)
            continue
        seat = person_seat.get(k)
        if seat is None:
            keys.append(UNSURE_SPEAKER)
            stats["リンク不能"] += 1
            continue
        m_at = mint_idx.get(seat)
        if m_at is None or i < m_at:
            keys.append(UNSURE_SPEAKER)
            stats["鋳造前"] += 1
        else:
            keys.append(uf.find(seat))
    return keys, {}, {"stats": dict(stats)}


def score_vs_gt(sess: Session, finals: list[str]) -> dict | None:
    """GT付きセッションの採点（全発話/実質=相槌除き/誤帰属/未確定）."""
    if not sess.gt_by_ms:
        return None
    pairs = []
    for d, pred in zip(sess.utts, finals, strict=True):
        code = sess.gt_by_ms.get(d["ms"])
        if code in ("S1", "S2", "S3"):
            pairs.append((pred, code, sess.is_bc(d)))
    if not pairs:
        return None
    acc, mapping = _gtlib.best_mapping([(p, g) for p, g, _ in pairs],
                                       ("S1", "S2", "S3"), unsure=UNSURE_SPEAKER)
    non_bc = [(p, g) for p, g, bc in pairs if not bc]
    ok = sum(1 for p, g in non_bc if mapping.get(p) == g)
    uns = sum(1 for p, _ in non_bc if p == UNSURE_SPEAKER)
    wrong = sum(1 for p, g in non_bc
                if p != UNSURE_SPEAKER and mapping.get(p, "__miss__") != g)
    n = len(non_bc)
    return {"acc_all": acc, "n_all": len(pairs), "n": n,
            "acc": ok / n, "unsure": uns / n, "wrong": wrong / n}


def unsure_rate(sess: Session, finals: list[str]) -> float:
    """相槌除き発話の未確定率（GT無しセッション用の主指標）."""
    vals = [p for d, p in zip(sess.utts, finals, strict=True) if not sess.is_bc(d)]
    return sum(1 for p in vals if p == UNSURE_SPEAKER) / max(len(vals), 1)


def fidelity(sess: Session, tmp: Path, max_speakers: int | None) -> tuple[float, int]:
    """現状シナリオの再現忠実度: 実ランの turns.jsonl の話者分割との一致率.

    diag の key 系列＋記録済みリネームを SessionState 実物へ流した結果が、
    実際に議事録へ出た話者分割を再現するかを見る。表示文字（参加者A/B/C）の
    採番順は本質でないため、最適1:1対応で突き合わせる（未確定は未確定同士で
    一致を要求＝安全側）。ここが低いと反実仮想の差分は信用できない
    （この数字を必ず添えて報告すること）。
    """
    keys, renames = scenario_base(sess)
    got = replay_seats(sess, keys, renames, max_speakers, tmp, disp=True)
    want_by_ms: dict[int, str] = {}
    for t in _gtlib.read_jsonl(ROOT / "transcripts" / f"{sess.name}.turns.jsonl"):
        want_by_ms.setdefault(t["ms"], t.get("speaker"))
    pairs = [(g, want_by_ms[d["ms"]]) for d, g in zip(sess.utts, got, strict=True)
             if want_by_ms.get(d["ms"]) is not None]
    if not pairs:
        return 0.0, 0
    live_labels = tuple(sorted({w for _, w in pairs if w != "未確定"}))
    acc, _ = _gtlib.best_mapping(pairs, live_labels, unsure="未確定")
    # 未確定同士の一致も再現とみなす（best_mapping は未確定を常に不一致に数える）
    both_unsure = sum(1 for g, w in pairs if g == "未確定" and w == "未確定")
    return acc + both_unsure / len(pairs), len(pairs)


def stage_utt(sessions: list[str], links_path: Path,
              th_b: float, tmp_root: Path) -> None:
    links = load_links(links_path)
    cols = ["現状", "案B即", "案B遅", "案A悲", "案A楽"]
    print(f"{'session':<20}{'指標':<6}" + "".join(f"{c:>7}" for c in cols)
          + "   備考")
    agg: dict[str, list[float]] = {}
    for name in sessions:
        sess = Session(name)
        max_sp = MAX_SPEAKERS.get(name, sess.max_speakers)
        scen = [scenario_base(sess), scenario_b(sess, links, th_b),
                scenario_b(sess, links, th_b, field="sim_full")]
        a_pess = scenario_a(sess, links, optimistic=False)
        a_opt = scenario_a(sess, links, optimistic=True)
        finals = [replay_seats(sess, k, r, max_sp,
                               tmp_root / name / str(i))
                  for i, (k, r) in enumerate(
                      [*scen, a_pess[:2], a_opt[:2]])]
        fid, _n_fid = fidelity(sess, tmp_root / name / "fid", max_sp)
        sc = [score_vs_gt(sess, f) for f in finals]
        if sc[0] is not None:
            for key, label in (("acc", "実質"), ("wrong", "誤帰属"),
                               ("unsure", "未確定")):
                note = (f"   n={sc[0]['n']} 再現{fid:.0%}"
                        if key == "acc" else "")
                print(f"{name:<20}{label:<6}"
                      + "".join(f"{s[key]:>7.1%}" for s in sc) + note)
                agg.setdefault(key, []).append(0.0)
                agg[key][-1] = 0.0
                for j, s in enumerate(sc):
                    agg.setdefault(f"{key}{j}", []).append(s[key])
        else:
            us = [unsure_rate(sess, f) for f in finals]
            print(f"{name:<20}{'未確定':<6}"
                  + "".join(f"{u:>7.1%}" for u in us)
                  + f"   (GT無し) 再現{fid:.0%}")
            for j, u in enumerate(us):
                agg.setdefault(f"gtless_unsure{j}", []).append(u)
        sys.stdout.flush()
    print()
    for key, label in (("acc", "実質"), ("wrong", "誤帰属"), ("unsure", "未確定")):
        vals = [agg.get(f"{key}{j}", []) for j in range(5)]
        if vals[0]:
            print(f"{'GT11本 平均':<20}{label:<6}"
                  + "".join(f"{sum(v)/len(v):>7.1%}" for v in vals))
    vals = [agg.get(f"gtless_unsure{j}", []) for j in range(5)]
    if vals[0]:
        print(f"{'実会話5本 平均':<20}{'未確定':<6}"
              + "".join(f"{sum(v)/len(v):>7.1%}" for v in vals))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True,
                   choices=["links", "linkops", "linksim", "utt"])
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--model", default="redimnet")
    p.add_argument("--out", default=None)
    p.add_argument("--th-b", type=float, default=0.45,
                   help="案Bのリンク閾値（鋳造時対称類似）")
    args = p.parse_args(argv)
    sessions = args.sessions or (GT_SESSIONS + REAL_SESSIONS)
    if args.stage == "links":
        out = Path(args.out or ROOT / "eval" / "phase0_links.jsonl")
        stage_links(sessions, Embedder(args.model), out)
    elif args.stage == "linkops":
        stage_linkops(Path(args.out or ROOT / "eval" / "phase0_links.jsonl"),
                      Embedder(args.model))
    elif args.stage == "linksim":
        stage_linksim(sessions, Embedder(args.model),
                      Path(args.out or ROOT / "eval" / "phase0_links.jsonl"))
    elif args.stage == "utt":
        import tempfile
        links_path = Path(args.out or ROOT / "eval" / "phase0_links.jsonl")
        stage_utt(sessions, links_path, args.th_b,
                  Path(tempfile.mkdtemp(prefix="phase0_seats_")))


if __name__ == "__main__":
    main()
