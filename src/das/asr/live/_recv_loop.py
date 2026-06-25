"""WebSocket受信ループ + flush処理."""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._session_state import SessionState
    from .stt import STTBackend

import contextlib

from ._constants import _BACKCHANNEL_RE, RESET, UNSURE_SPEAKER, fmt_ts
from ._ui import _print_line


class RecvLoop:
    """STTからのトークンストリームを処理し、発話を確定(flush)してrecordsに追加する.

    内部トークン形式（STTBackend.parse_messageの出力）を受け取り、
    声紋判定・エコー除去・records追加・ファイル保存を行う。
    STTプロバイダには依存しない（バックエンドが変換済みトークンを供給）。
    """

    _FLUSH_SILENCE = 1.6      # 最終トークンが来ない無音がこの秒数続いたら確定（自前の区切り）
    _FLUSH_SOFT_CHARS = 500   # この文字数を超えたら文の切れ目でflush
    _FLUSH_HARD_CHARS = 1000  # この文字数を超えたら問答無用で強制flush

    def __init__(self, state: SessionState, args, backend: STTBackend):
        self.state = state
        self.args = args
        self.backend = backend
        self.cur_speaker = None
        self.cur_text = ""
        self.cur_ms: int | None = None
        self.cur_end: int | None = None
        self.cur_last_token_time: float = time.monotonic()
        self.recent_segs: list[tuple] = []

    def overlaps_other(self, start, end, label) -> bool:
        if start is None or end is None:
            return False
        return any(lbl != label and min(e, end) - max(s, start) > 0
                   for s, e, lbl in self.recent_segs)

    def flush(self):
        from das.asr.live import ON_UTTERANCE

        s = self.state
        if not self.cur_text.strip():
            self.cur_text = ""
            self.cur_ms = None
            self.cur_end = None
            self.cur_last_token_time = time.monotonic()
            return
        label = str(self.cur_speaker)
        tracker = s.tracker
        agent = s.agent
        partner = s.partner
        # 相槌（「はい」等）は声紋の人物確定に使わず、UIでも薄く折りたためるよう印を付ける
        _is_backchannel = bool(_BACKCHANNEL_RE.match(self.cur_text.strip()))
        if tracker is not None:
            if self.cur_ms is not None and self.cur_end is not None and self.cur_end > self.cur_ms:
                with s.buf_lock:
                    abs_start = self.cur_ms * 32
                    abs_end = self.cur_end * 32
                    rel_start = max(abs_start - s.pcm_buf_offset, 0)
                    rel_end = max(abs_end - s.pcm_buf_offset, 0)
                    seg = bytes(s.pcm_buf[rel_start: rel_end])
                wav = np.frombuffer(seg, dtype="<i2").astype(np.float32) / 32768.0
            else:
                wav = np.zeros(0, dtype=np.float32)
            # 登録は声ごとの累積文字数で判定するので、この発話の文字数を渡す
            sp_id = tracker.classify(
                wav, self.cur_speaker,
                overlapped=self.overlaps_other(self.cur_ms, self.cur_end, label),
                count=not _is_backchannel, chars=len(self.cur_text.strip()))
            # --- 声紋ベースのAIエコー除去 ---
            if (sp_id is not None
                    and sp_id.startswith("__") and sp_id.endswith("__")):
                if self.args.vp_debug:
                    _print_line(f"# AI声紋エコー除去: sp={sp_id}"
                                f" ({self.cur_text.strip()[:40]}...)")
                self.cur_text = ""
                self.cur_ms = None
                self.cur_end = None
                return
            d = tracker.last
            rec_extra = {}
            if d and d["kind"] == "補正":
                note = (f"声紋でラベル{d['label']}の取り違えを修正"
                        f"（類似{d['sim']:.2f}、放置なら{s.disp_name(d['prev'])}の発言になっていた）")
                rec_extra = {"vp": "補正", "note": note}
                _print_line(f"# ⚡補正: {note}")
            elif d and d["kind"] == "自動登録":
                if d["rename"]:
                    s.rekey(*d["rename"])
                s.add_sys(self.cur_ms, f"この声を「{d['name']}」として追跡開始"
                                       f"（実名にするには {d['label']}=名前）")
                _print_line(f"# この声を「{d['name']}」として追跡します"
                            f"（実名にするには {d['label']}=名前 と入力）")
            elif d and d["kind"] == "合流":
                if d["rename"]:
                    s.rekey(*d["rename"])
                if self.args.vp_debug:
                    _print_line(f"# 合流: ラベル{d['label']}→{d['name']}")
            elif self.args.vp_debug and d:
                extra = f" 類似{d['sim']:.2f}({d['name']})" if "sim" in d else ""
                _print_line(f"# vp判定[{d['kind']}]{extra}")
        else:
            sp_id = "#" + str(self.cur_speaker)
            rec_extra = {}
        # --- テキスト類似度エコー判定（安全網） ---
        for _src_name, _src in [("agent", agent), ("partner", partner)]:
            if _src is None:
                continue
            if _src_name == "agent" and not _src.in_echo_window:
                continue
            sim = _src._best_similarity(self.cur_text)
            if sim > 0.35:
                if self.args.vp_debug:
                    _print_line(f"# テキスト安全網エコー除去({_src_name})"
                                f" sim={sim:.2f}: sp={sp_id}"
                                f" ({self.cur_text.strip()[:40]}...)")
                self.cur_text = ""
                self.cur_ms = None
                self.cur_end = None
                return
        if self.cur_ms is not None and self.cur_end is not None:
            self.recent_segs.append((self.cur_ms, self.cur_end, label))
            del self.recent_segs[:-12]
        if tracker is not None and tracker.last is not None:
            try:
                with open(s.diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ms": self.cur_ms, "end": self.cur_end, "label": label,
                                        "key": sp_id, **tracker.last},
                                       ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        if _is_backchannel:
            # 相槌は、話している人とは別人の可能性が高い（Aの話中にBが「はい」）。
            # 直前の人に追従させず未確定にする。bcフラグでUIでは薄く折りたたむ。
            sp_id = UNSURE_SPEAKER
            rec_extra["bc"] = True
        with s.state_lock:
            s.records.append({"ms": self.cur_ms, "end_ms": self.cur_end,
                              "speaker": sp_id, "text": self.cur_text.strip(),
                              **rec_extra})
            c = s.color_of(sp_id)
        if ON_UTTERANCE is not None:
            with contextlib.suppress(Exception):
                ON_UTTERANCE(s.disp_name(sp_id), self.cur_text.strip())
        _print_line(f"{c}[{fmt_ts(self.cur_ms)}] {s.disp_name(sp_id)}{RESET}: {self.cur_text.strip()}")
        s.save()
        self.cur_text = ""
        self.cur_ms = None
        self.cur_end = None
        self.cur_last_token_time = time.monotonic()

    def run(self, ws):
        """WebSocket受信ループのメイン.

        stop（終了）または reset_requested（STT作り直し）で正常に抜ける。
        後者の場合、呼び出し側（run_session）が新しい ws を張り直す。
        """
        args = self.args
        try:
            while not self.state.stop.is_set() and not self.state.reset_requested.is_set():
                try:
                    raw = ws.recv()
                except Exception:
                    # 停止/作り直しで ws が閉じられた場合は正常終了扱い
                    if self.state.stop.is_set() or self.state.reset_requested.is_set():
                        break
                    raise
                res = self.backend.parse_message(json.loads(raw), args.lang)
                if res.get("error_code") is not None:
                    _print_line(f"# エラー: {res['error_code']} - {res.get('error_message')}")
                    break
                partial = ""
                partial_sp = self.cur_speaker
                for token in res.get("tokens", []):
                    text = token.get("text") or ""
                    if text == "<end>":
                        self.flush()
                        continue
                    if not text:
                        continue
                    if token.get("is_final"):
                        sp = token.get("speaker")
                        if sp != self.cur_speaker:
                            self.flush()
                            self.cur_speaker = sp
                        if self.cur_ms is None:
                            self.cur_ms = token.get("start_ms")
                        if token.get("end_ms") is not None:
                            self.cur_end = token["end_ms"]
                        self.cur_text += text
                        self.cur_last_token_time = time.monotonic()
                    else:
                        partial += text
                        partial_sp = token.get("speaker") or partial_sp
                # --- 強制flush ---
                # 区切りは「話者変化」(上)＋「無音」＋文字数で行う。エンドポイント検出OFF時の
                # 主な区切りはこの無音flush。Sonioxの確定タイミングには影響しない
                # （既に is_final になったトークンをまとめるだけ）。
                if self.cur_text:
                    clen = len(self.cur_text)
                    idle = time.monotonic() - self.cur_last_token_time
                    if (idle > self._FLUSH_SILENCE
                            or clen > self._FLUSH_HARD_CHARS
                            or (clen > self._FLUSH_SOFT_CHARS
                                and self.cur_text.rstrip()[-1:] in "。？！.?!\n")):
                        self.flush()
                self.state.show_partial(partial_sp if partial else self.cur_speaker,
                                        self.cur_text + partial)
                if res.get("finished"):
                    self.flush()
                    _print_line("# 終了")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.flush()
