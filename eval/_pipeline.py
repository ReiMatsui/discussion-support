"""記録から「今日の実装」を再現して採点する、ただ1つの実装.

**なぜ1つにまとめるのか**: 同じ再現が11本のスクリプトに書き写されていた。
書き写しは必ずずれる。実際にずれた:

  - `sortformer_compare` が古い `final_key`（記録時の判定）をそのまま「現行」
    として比べ、現行 54.4% 対 Sortformer 87.7% という**誤った結論**を出した。
    本当は 89.9% 対 87.7% だった（handoff §29）
  - `seat_pick_variants` と `seat_query_context` だけ「蓄積中の門番」を通さず
    採点していた。同じ「今日の実装」を名乗る数字が2種類あった

採点の規則を1箇所に置けば、こういう食い違いは構造的に起きない。新しい調査
スクリプトは、この規則を**呼ぶ**か、意図的に外すかを明示することになる。

規則（本番 `_recv_loop.flush` と `_seat_audio` に対応）:

    1. 相槌は分母から外す（聞き手が打つので話者が別人になりやすい）
    2. 蓄積中で裏付けの無い帰属は未確定に落とす（§27.11）
    3. 根拠がSTTラベルだけの kind（ラベル不純・ラベル継続）は席の音声で決め直す
    4. 上流が決めていたのに席上限で落ちた発話も席の音声で決め直す（§27.8）
    5. 席の参照は「声紋層が高信頼だった発話」だけから作る（§27.9）
    6. 予定表の時刻が来たら、控えてある声紋を今の参照で貼り直す（§28）
    7. 1秒未満で1位と2位が僅差なら、寄せずに未確定にする（§36）
    8. 席の割当ての声紋だけ、声紋層より大きいモデルを使う（§38）
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._attribution import _VOICEPRINT_RELIABLE_KINDS  # noqa: E402
from das.asr.live._constants import (  # noqa: E402
    RETRO_INTERVAL_SEC,
    RETRO_SCHEDULE_SEC,
    SEAT_AUDIO_MIN_REF_SEC,
    SR,
    UNSURE_SPEAKER,
)
from das.asr.live._recv_loop import _LABEL_ONLY_KINDS  # noqa: E402
from das.asr.live._seat_audio import (  # noqa: E402
    SeatAudio,
    declines_short,
    seat_embedder,
)

__all__ = ["apply_schedule", "current_keys", "gt_rows", "pick_nearest",
           "read_wav", "replay_seats", "resolved_key", "score"]


# ---------------------------------------------------------------- 素材


def read_wav(path: Path) -> np.ndarray:
    """録音を float32 の波形で読む（mono/16kHz 以外は測らずに止める）."""
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getframerate() != SR:
            raise SystemExit(f"# {path} は mono/{SR}Hz ではない")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def gt_rows(run: str, *, align: str = "time") -> list[tuple[dict, str]] | None:
    """採点対象の (発話, GTコード) を時刻順で返す（相槌は除く）.

    相槌を分母に入れると実会話の未確定率が3倍に見える。§28.14 で実際に
    誤った報告をしたので、除外はここに固定する。

    `align` は正解の割り当て方:

      "time" 時間の重なり（従来）。8割を一人が占めるときだけ正解を付ける。
             重なりの多い場面が丸ごと落ちる。
      "text" 文章の一致（`_textgt`）。「この一文は誰のものか」で決めるので
             重なっていても答えが出る。落ちるのは相づちだけ。
    """
    loaded = dec.load_run(run)
    if loaded is None:
        return None
    utts, code_by_ms = loaded
    if align == "text":
        import _textgt
        by_text = _textgt.codes_by_ms(run, utts)
        if by_text is None:
            return None
        code_by_ms = by_text
    rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
    rows = [(u, c) for u, c in rows if c in dec.GT_CODES
            and not dec._BACKCHANNEL_RE.match(str(u["_text"]).strip())]
    rows.sort(key=lambda r: int(r[0]["ms"]))
    return rows or None


# ---------------------------------------------------------------- 採点


def score(pairs: list[tuple[str, str]]) -> tuple[float, float, float, int]:
    """(正解率, 誤帰属率, 未確定率, 件数) を返す.

    最適1:1対応を取るのは、システム側のキー（人物1 等）と GT のコード（S1 等）
    に共通の名前空間が無いため。未確定は対応の候補から外れ、常に不正解として
    分母にだけ入る。
    """
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    n = len(pairs)
    good = sum(1 for f, c in pairs if m.get(f) == c)
    uns = sum(1 for f, _ in pairs if f == UNSURE_SPEAKER)
    return good / n, (n - good - uns) / n, uns / n, n


# ------------------------------------------------ 記録1件に規則を当てる


def resolved_key(u: dict, pick: str | None = None, *, cap: bool = True) -> str:
    """記録された発話1件に、今日の規則を当てた最終キーを返す.

    `pick` はその発話を席の音声で決め直した結果（無ければ None）。席で
    決め直す条件もここが持つ——呼び出し側が独自に判断すると、また2種類の
    「今日の実装」ができる。

    `cap=False` は**参加人数を決めていない**場合（統一席ルールを掛けない）。
    diag は席上限を掛ける前のキー（`key`）と後（`final_key`）の両方を持って
    いるので、記録から再現できる。上限が無ければ「上限で落ちた発話」も存在
    しないので、それを拾う経路（§27.8）ごと消える。
    """
    cur = str(u.get("key")) if not cap else str(
        u["final_key"] if u.get("final_key") is not None else u.get("key"))
    kind = u.get("kind")
    # 蓄積中の門番（§27.11）。人数の情報には依存しないので上限の有無に関わらず効く。
    if cur != UNSURE_SPEAKER and kind == "蓄積中" and not dec.endorsed(u):
        cur = UNSURE_SPEAKER
    # 根拠がラベルだけの kind は席の音声で決め直す（§27.12）
    if kind in _LABEL_ONLY_KINDS and pick:
        return pick
    if cur != UNSURE_SPEAKER:
        return cur
    # 上流は決めていたのに席上限で落ちた分（§27.8）
    if cap and str(u.get("key")) != UNSURE_SPEAKER and pick:
        return pick
    return cur


def is_revisable(u: dict, *, cap: bool = True, seats: bool = True) -> bool:
    """席の音声で決め直す対象か（`resolved_key` が `pick` を使う条件と同じ）.

    `seats=False` は席の割当てそのものを使わない条件。閉集合の割当てが成り
    立つ根拠は「参加人数の設定上そこに新しい人は入らない」ことなので、人数を
    決めていないなら本来この仕組みは正当化できない（§27）。人数の情報を一切
    使わない場合の成績を測るときに使う。
    """
    if not seats:
        return False
    kind = u.get("kind")
    if kind in _LABEL_ONLY_KINDS:
        return True
    if not cap:
        return False   # 上限が無ければ「上限で落ちた発話」も無い
    cur = str(u["final_key"] if u.get("final_key") is not None else u.get("key"))
    if cur != UNSURE_SPEAKER and kind == "蓄積中" and not dec.endorsed(u):
        cur = UNSURE_SPEAKER
    return cur == UNSURE_SPEAKER and str(u.get("key")) != UNSURE_SPEAKER


def pick_nearest(emb, refs: dict, st: dict | None = None) -> str | None:
    """席の参照のうち最も似ている1人（席が2つ未満なら選ばない）.

    類似度そのものの下限は課さない。閉集合の割当てなので「誰でもない」という
    選択肢は適用条件（席上限で落ちた発話に限る）が既に排除している（§27.7）。

    ただし**短くて僅差なら未確定を返す**（§36）。判定は本番と同じ述語
    （`_seat_audio.declines_short`）を呼ぶ——ここに条件を書き写すと、また
    「今日の実装」を名乗る数字が2種類できる。`st` を渡さない呼び出しは
    長さが分からないので、この棄権は掛からない。
    """
    if emb is None or len(refs) < 2:
        return None
    ranked = sorted(((float(np.dot(emb, v)), k) for k, v in refs.items()),
                    reverse=True)
    picked = (ranked[0][1], ranked[0][0], ranked[0][0] - ranked[1][0])
    dur = None
    if st is not None:
        u = st["utt"]
        dur = max(0, int(u.get("end") or u["ms"]) - int(u["ms"]))
    return UNSURE_SPEAKER if declines_short(picked, dur) else picked[0]


# ------------------------------------------------------------ 再生


def replay_seats(run: str, vp, *, wav_path: Path | None = None,
                 align: str = "time", query=None, cap: bool = True,
                 seats: bool = True) -> dict | None:
    """席の参照の推移と、貼り直せる発話の声紋を1回だけ計算する.

    席の参照は「高信頼で確定した発話」だけから作られ、その集合は予定表に
    依存しない。したがって推移を保存しておけば、どの予定表でも再利用できる
    （条件を増やしても埋め込みの計算は増えない）。

    `cap` / `seats` は「参加人数を決めているか」の条件（`resolved_key` /
    `is_revisable` 参照）。人数の情報にどれだけ寄りかかっているかを測るための
    入口で、既定は本番と同じ「決めている」。

    `query` は「席と比べる音声を何にするか」の差し込み口で、全発話について
    ``query(発話, 音声, 決め直す対象か) -> {名前: 音声}`` の形で呼ばれる。
    複数案を1つの辞書で返せるようにしてあるのは、案ごとに流し直すと席の参照
    まで作り直しになり、埋め込みの計算が案の数だけ増えるため。既定は現行の
    「その発話の音声だけ」（名前は空文字）。
    """
    rows = gt_rows(run, align=align)
    wav_path = wav_path or ROOT / "transcripts" / f"{run}.wav"
    if rows is None or not wav_path.exists():
        return None
    pcm = read_wav(wav_path)
    t0 = int(rows[0][0]["ms"])
    # 席の埋め込み器の選び方も本番と同じ関数に任せる（§38）。ここで
    # `SeatAudio(vp)` と書くと、本番だけ b5・再現は b2 という食い違いになる。
    seat = SeatAudio(vp, embedder=seat_embedder(vp))
    steps = []
    for u, code in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        revisable = is_revisable(u, cap=cap, seats=seats)
        base = resolved_key(u, cap=cap)
        want = (query(u, wav, revisable) if query
                else ({"": wav} if revisable else {}))
        embs = {k: seat.embed(v) for k, v in (want or {}).items()}
        if not revisable and base != UNSURE_SPEAKER \
                and u.get("kind") in _VOICEPRINT_RELIABLE_KINDS:
            seat.observe(base, wav)
        refs = {k: v for k, v in seat._embeddings.items()
                if seat._seconds.get(k, 0.0) >= SEAT_AUDIO_MIN_REF_SEC}
        steps.append({"ms": a, "elapsed": (a - t0) / 1000.0, "code": code,
                      "base": base, "revisable": revisable,
                      "emb": embs.get(""), "embs": embs,
                      "refs": dict(refs), "utt": u})
    return {"run": run, "steps": steps}


def apply_schedule(steps: list[dict], schedule=RETRO_SCHEDULE_SEC,
                   interval: float = RETRO_INTERVAL_SEC, *,
                   pick=None, name: str = "") -> list[str]:
    """予定表どおりに遡及訂正を掛け、発話ごとの最終キーを返す（§28）.

    貼り直しは保存済みの声紋と席の参照の内積だけで、埋め込みの計算は要らない。
    間隔を詰めない理由は計算量ではなく、表示が頻繁に書き換わること（UX）だけ。

    `pick` は寄せ先の選び方で、``pick(声紋, 席の参照, その発話の記録)`` の形。
    第3引数を渡すのは、長さや kind で条件を変える案（短い発話にだけ棄権を
    許す等）を、選び方の差し替えだけで測れるようにするため。`name` は
    `replay_seats` の `query` が付けた声紋の名前。どちらも**最初の判定と
    貼り直しの両方**に同じものが使われる——片方だけ替えると、その差が案の
    効果に混ざる。
    """
    pick = pick or pick_nearest
    final: list[str] = []
    remembered: list[int] = []
    idx = 0
    next_at = schedule[0] if schedule else interval
    for i, st in enumerate(steps):
        cur = st["base"]
        if st["revisable"]:
            got = pick(st["embs"].get(name), st["refs"], st)
            cur = got if got is not None else cur
            remembered.append(i)
        final.append(cur)
        if st["elapsed"] >= next_at:
            idx += 1
            next_at = (schedule[idx] if idx < len(schedule)
                       else st["elapsed"] + interval)
            for j in remembered:
                got = pick(steps[j]["embs"].get(name), st["refs"],
                           steps[j])
                if got is not None:
                    final[j] = got
    return final


def current_keys(run: str, vp) -> dict[int, str] | None:
    """**今日の実装**で決まる最終キーを ms -> キー で返す.

    diag の `final_key` は記録時の判定なので、それをそのまま「現行」として
    比べると古い比較を繰り返すことになる（§29 で実際にそうなった）。本番の
    予定表・本番のクラスで計算し直す。
    """
    data = replay_seats(run, vp)
    if data is None:
        return None
    final = apply_schedule(data["steps"])
    return {int(st["ms"]): f
            for st, f in zip(data["steps"], final, strict=True)}
