"""main()内の共有状態を集約するコンテナ."""
from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from collections.abc import Callable

from ._constants import (
    _PROACTIVITY_DEFAULT,
    _PROACTIVITY_PROFILES,
    AGENT_SPEAKER,
    AGENT_VOICES,
    CLEAR_LINE,
    DIM,
    HTML_PALETTE,
    HTML_TMPL,
    PALETTE,
    RESET,
    SR,
    fmt_ts,
)
from ._participation import participation_stats
from ._voice_profiles import VoiceProfiles
from .agents._partner import ConversationPartner
from .agents._realtime import RealtimeAgent
from .agents._simulator import DiscussionSimulator


class SessionState:
    """main()内の共有状態を集約するコンテナ.

    巨大だった main() のクロージャ変数をインスタンス属性に集約し、
    ヘルパーメソッドとして外部からアクセス可能にする。
    """

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def __init__(self, *, args, started, out_path, html_path, diag_path,
                 turns_path, wav_path, tracker=None, serve=True):
        self.args = args
        self.started = started
        self.out_path = out_path
        self.html_path = html_path
        self.diag_path = diag_path
        self.turns_path = turns_path
        self.wav_path = wav_path
        self._serve = serve

        # 発話記録
        self.names: dict[str, str] = {}
        self.colors: dict[str, str] = {}
        self.records: list[dict] = []
        self.state_lock = threading.Lock()

        # 声紋
        self.tracker: VoiceProfiles | None = tracker

        # AI
        self.agent: RealtimeAgent | None = None
        self.partner: ConversationPartner | None = None
        self.simulator: DiscussionSimulator | None = None

        # 論点
        self.topics: list[dict] = []
        self.topics_lock = threading.Lock()
        self.topic_cursor = 0
        self._TOPIC_WINDOW = 10
        self._TOPIC_TRIGGER = 5
        self.drift_cursor = 0
        # 脱線検出→介入トリガーの受け渡しキュー（R2: トリガー経路の単一化）。
        # _run_drift_checker が積み、_run_agent_worker が裁定して trigger する。
        self.drift_requests: queue.Queue[str] = queue.Queue()
        # 参加度の声かけ要求キュー（S4）。対象話者の表示名を積む。
        self.invite_requests: queue.Queue[str] = queue.Queue()
        # 積極性プロファイル（S5）。bootstrapで --proactivity から上書きされる。
        self.proactivity: dict = dict(_PROACTIVITY_PROFILES[_PROACTIVITY_DEFAULT])
        # UIからの停止フック（F1）。run_sessionが「stopを立ててwsを閉じる」関数を設定する。
        self.request_stop: Callable[[], None] | None = None

        # PCMバッファ
        self.pcm_buf = bytearray()
        self.pcm_buf_offset = 0
        self.pcm_total_bytes = 0
        self._PCM_KEEP_BYTES = SR * 2 * 120
        self.buf_lock = threading.Lock()
        self.pcm_file = None  # IO[bytes] | None

        # 制御
        self.stop = threading.Event()
        self.audio_q: queue.Queue[bytes | None] = queue.Queue()

        # エージェントワーカー状態
        self._last_utt_time = [time.monotonic()]
        self._was_in_echo = [False]
        self.agent_cursor = 0

    # ------------------------------------------------------------------
    # 表示ヘルパー
    # ------------------------------------------------------------------
    def disp_name(self, key) -> str:
        key = str(key)
        if key in self.names:
            return self.names[key]
        return f"話者{key[1:]}" if key.startswith("#") else key

    def key_for_label(self, sp) -> str:
        sp = str(sp)
        if self.tracker is not None and sp in self.tracker.sp_map:
            return self.tracker.sp_map[sp]
        return "#" + sp

    def color_of(self, key) -> str:
        key = str(key)
        if key not in self.colors:
            self.colors[key] = PALETTE[len(self.colors) % len(PALETTE)]
        return self.colors[key]

    def rekey(self, old: str, new: str):
        """表示キーの付け替え: recordsと色を一括移行."""
        with self.state_lock:
            for r in self.records:
                if r.get("speaker") == old:
                    r["speaker"] = new
            if old in self.colors:
                self.colors.setdefault(new, self.colors.pop(old))

    def add_sys(self, ms, text: str):
        """システムイベントを議事録のタイムラインに残す."""
        with self.state_lock:
            self.records.append({"ms": ms, "sys": text})

    # ------------------------------------------------------------------
    # UI(API)向けの状態スナップショット（F1）
    # ------------------------------------------------------------------
    def session_mode(self) -> str:
        """現在のセッションモード: transcribe / converse / facilitate.

        - transcribe : 議事録のみ（エージェントoff or 無し）
        - converse   : AIと会話（パートナー有り＋ファシリテーター）
        - facilitate : 人間同士に介入（ファシリテーター単体）
        """
        if self.agent is None or not self.agent.enabled:
            return "transcribe"
        if self.partner is not None:
            return "converse"
        return "facilitate"

    def api_snapshot(self) -> dict:
        """UI(API)向けの現在状態（JSON化可能なdict）を返す."""
        with self.state_lock:
            raw = list(self.records)
            records = []
            for r in raw:
                if "sys" in r:
                    records.append({"type": "sys", "ms": r.get("ms"),
                                    "text": r["sys"]})
                elif "speaker" in r:
                    records.append({
                        "type": "utt",
                        "ms": r.get("ms"), "end_ms": r.get("end_ms"),
                        "speaker": self.disp_name(r["speaker"]),
                        "text": r.get("text", ""),
                        "corrected": r.get("vp") == "補正",
                    })
            speakers = list(dict.fromkeys(
                self.disp_name(r["speaker"]) for r in raw if "speaker" in r))
        with self.topics_lock:
            topics = [{"topic": t.get("topic", ""), "speaker": t.get("speaker", "")}
                      for t in self.topics]
        stats = participation_stats(raw, exclude_speakers=(AGENT_SPEAKER, "パートナー"))
        participation = [
            {"speaker": self.disp_name(sp),
             "time_share": round(d["time_share"], 3), "turns": d["turns"]}
            for sp, d in stats.items()
        ]
        agent = None
        if self.agent is not None:
            agent = {"enabled": self.agent.enabled, "mode": self.agent.mode,
                     "voice": self.agent.voice}
        return {
            "mode": self.session_mode(),
            "running": not self.stop.is_set(),
            "started": self.started.strftime("%Y-%m-%d %H:%M"),
            "speakers": speakers,
            "records": records,
            "topics": topics,
            "participation": participation,
            "agent": agent,
        }

    def seed_topic(self, topic: str | None, speaker: str = "議題"):
        """明示的な議題を脱線検出の基準論点としてシードする（Fix 8）.

        debate/simulate モードのように議題が分かっている場合、論点抽出LLMの
        成否を待たずに最初から脱線検出を効かせるための初期基準を入れる。
        既に論点があれば何もしない（抽出済みを優先）。
        """
        if not topic:
            return
        with self.topics_lock:
            if not self.topics:
                self.topics.append({"topic": topic, "speaker": speaker})

    def key_of(self, tok: str) -> str:
        """コマンド引数を表示キーへ: 人物名はそのまま、数字はそのラベルの現在の表示先."""
        if self.tracker is not None:
            if tok in self.tracker.profiles:
                return tok
            if tok in self.tracker.sp_map:
                return self.tracker.sp_map[tok]
        return "#" + tok

    def show_partial(self, sp, text: str):
        if not text.strip():
            sys.stdout.write(CLEAR_LINE)
        else:
            cols = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            line = f"{self.disp_name(self.key_for_label(sp))}: {text.strip()}"
            sys.stdout.write(CLEAR_LINE + DIM + line[-(cols - 2):] + RESET)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # 出力
    # ------------------------------------------------------------------
    def write_md(self, recs=None, path=None):
        with self.state_lock:
            rs = self.records if recs is None else recs
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            lines = [
                f"# 議事録 {self.started.strftime('%Y-%m-%d %H:%M')}",
                "",
                "話者: " + (", ".join(self.disp_name(s) for s in speakers) or "（未検出）"),
                "",
            ]
            for r in rs:
                if "sys" in r:
                    lines.append(f"> [{fmt_ts(r['ms'])}] {r['sys']}")
                    continue
                mark = " ⚡" if r.get("vp") == "補正" else ""
                lines.append(f"- **[{fmt_ts(r['ms'])}] {self.disp_name(r['speaker'])}{mark}**: {r['text']}")
            dst = path or self.out_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, dst)

    def write_html(self, live: bool = True, recs=None, path=None, status=None):
        import html as _html
        with self.state_lock:
            rs = self.records if recs is None else recs
            parts = []
            for r in rs:
                if "sys" in r:
                    parts.append(f'<div class="sys">⚙ {_html.escape(r["sys"])}</div>')
                    continue
                sp = str(r["speaker"])
                self.color_of(sp)
                idx = list(self.colors).index(sp)
                c = HTML_PALETTE[idx % len(HTML_PALETTE)]
                badge = ""
                if r.get("vp") == "補正":
                    note = _html.escape(r.get("note", ""))
                    badge = f'<span class="badge" title="{note}">⚡声紋補正</span>'
                parts.append(
                    f'<div class="u"><span class="ts">{fmt_ts(r["ms"])}</span>'
                    f'<span class="who" style="color:{c}">{_html.escape(self.disp_name(sp))}</span>'
                    f'{_html.escape(r["text"])}{badge}</div>'
                )
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            sp_tags = []
            for s in speakers:
                dn = _html.escape(self.disp_name(s))
                idx_s = list(self.colors).index(s) if s in self.colors else 0
                c = HTML_PALETTE[idx_s % len(HTML_PALETTE)]
                is_renameable = self._serve and self.tracker is not None and not s.startswith("#")
                if is_renameable:
                    lbl = s
                    for _l, _k in self.tracker.sp_map.items():
                        if _k == s:
                            lbl = _l
                            break
                    is_anon = re.match(r"^人物\d+$", s)
                    ph = "名前" if is_anon else "新しい名前"
                    sp_tags.append(
                        f'<div class="speaker-tag">'
                        f'<div class="speaker-name"><span class="dot" style="background:{c}"></span>{dn}</div>'
                        f'<div class="rename-row">'
                        f'<input class="rename-input" placeholder="{ph}" data-label="{_html.escape(lbl)}">'
                        f'<button class="rename-btn" onclick="rename(this)">登録</button>'
                        f'</div></div>')
                else:
                    sp_tags.append(
                        f'<div class="speaker-tag">'
                        f'<div class="speaker-name"><span class="dot" style="background:{c}"></span>{dn}</div>'
                        f'</div>')
            if sp_tags:
                speaker_panel = ('<div class="sidebar"><p class="sidebar-title">この会議の話者</p>'
                                 '<div class="speaker-panel">' + ''.join(sp_tags) + '</div></div>')
            else:
                speaker_panel = ''
            profile_panel = ''
            if self._serve and self.tracker is not None:
                all_names = self.tracker.all_profile_names()
                if all_names:
                    active_names = set(self.tracker.active_profile_names())
                    items = []
                    for n in all_names:
                        cls = 'profile-item active' if n in active_names else 'profile-item'
                        items.append(
                            f'<div class="{cls}" data-name="{_html.escape(n)}" '
                            f'onclick="toggleProfile(this)">'
                            f'<span class="profile-toggle"></span>'
                            f'{_html.escape(n)}</div>')
                    profile_panel = ('<div class="profile-section">'
                                     '<p class="sidebar-title">プロファイル</p>'
                                     + ''.join(items) + '</div>')
            stats_panel = ''
            talk_rs = [r for r in rs if "speaker" in r and r.get("text")]
            if talk_rs:
                sp_dur: dict[str, float] = {}
                sp_chars: dict[str, int] = {}
                sp_turns: dict[str, int] = {}
                for r in talk_rs:
                    s = r["speaker"]
                    ms, end = r.get("ms"), r.get("end_ms")
                    dur = (end - ms) / 1000.0 if ms is not None and end is not None and end > ms else 0.0
                    sp_dur[s] = sp_dur.get(s, 0.0) + dur
                    sp_chars[s] = sp_chars.get(s, 0) + len(r["text"])
                    sp_turns[s] = sp_turns.get(s, 0) + 1
                total_dur = sum(sp_dur.values()) or 1.0
                total_chars = sum(sp_chars.values()) or 1
                total_turns = sum(sp_turns.values()) or 1
                ranked = sorted(sp_dur.keys(), key=lambda s: sp_dur[s], reverse=True)

                def _bar_rows(data, total, unit=""):
                    rows = []
                    for s in ranked:
                        v = data.get(s, 0)
                        pct = v / total * 100 if total else 0
                        idx_s = list(self.colors).index(s) if s in self.colors else 0
                        c = HTML_PALETTE[idx_s % len(HTML_PALETTE)]
                        dn = _html.escape(self.disp_name(s))
                        short = dn[:2] if len(dn) > 3 else dn
                        rows.append(
                            f'<div class="stats-row">'
                            f'<span class="stats-name" title="{dn}">{short}</span>'
                            f'<div class="stats-bar-bg">'
                            f'<div class="stats-bar" style="width:{pct:.0f}%;background:{c}"></div>'
                            f'</div>'
                            f'<span class="stats-pct">{pct:.0f}%</span>'
                            f'</div>')
                    return ''.join(rows)

                groups = []
                if total_dur > 0.5:
                    groups.append('<div class="stats-group">'
                                  '<div class="stats-label">発話時間</div>'
                                  + _bar_rows(sp_dur, total_dur) + '</div>')
                groups.append('<div class="stats-group">'
                              '<div class="stats-label">文字数</div>'
                              + _bar_rows(sp_chars, total_chars) + '</div>')
                groups.append('<div class="stats-group">'
                              '<div class="stats-label">発話回数</div>'
                              + _bar_rows(sp_turns, total_turns) + '</div>')
                stats_panel = ('<div class="stats-section">'
                               '<p class="sidebar-title">発言量</p>'
                               + ''.join(groups) + '</div>')
            topics_panel = ''
            with self.topics_lock:
                if self.topics:
                    items = []
                    for t in self.topics:
                        tt = _html.escape(t.get("topic", ""))
                        ts = _html.escape(t.get("speaker", ""))
                        items.append(f'<div class="topic-item">'
                                     f'<div class="topic-text">{tt}</div>'
                                     f'<div class="topic-by">{ts}</div></div>')
                    topics_panel = ('<div class="topics-section">'
                                   '<p class="sidebar-title">論点</p>'
                                   + ''.join(items) + '</div>')
            agent_panel = ''
            if self.agent is not None:
                cur_mode = self.agent.mode
                if self.agent._connected:
                    conn = '接続中'
                elif self.agent._conn_error:
                    conn = f'エラー: {_html.escape(self.agent._conn_error)}'
                else:
                    conn = '未接続'
                mode_btns = []
                for m, lbl in [("off", "OFF"), ("facilitator", "進行役"),
                               ("conversation", "会話")]:
                    cls = "agent-mode-btn active" if m == cur_mode else "agent-mode-btn"
                    mode_btns.append(f'<button class="{cls}" data-mode="{m}" '
                                     f'onclick="setAgentMode(this)">{lbl}</button>')
                voice_opts = []
                for v in AGENT_VOICES:
                    sel = 'selected' if v == self.agent.voice else ''
                    voice_opts.append(f'<option value="{v}" {sel}>{v}</option>')
                trigger_val = self.agent.trigger_n
                agent_panel = (
                    f'<div class="agent-section" data-mode="{cur_mode}">'
                    f'<div class="agent-header">'
                    f'<span class="agent-label">🤖 AI Agent</span>'
                    f'<span class="agent-conn">{conn}</span>'
                    f'</div>'
                    f'<div class="agent-modes">{"".join(mode_btns)}</div>'
                    f'<div class="agent-opts">'
                    f'<label class="agent-opt-label">声'
                    f'<select class="agent-select" onchange="setAgentVoice(this)">'
                    f'{"".join(voice_opts)}</select></label>'
                    f'<label class="agent-opt-label agent-trigger-row">'
                    f'間隔 <input type="number" class="agent-num" value="{trigger_val}" '
                    f'min="1" max="50" onchange="setAgentTrigger(this)">発話'
                    f'</label>'
                    f'</div></div>')
            doc = HTML_TMPL.format(
                refresh='<meta http-equiv="refresh" content="2">' if live else "",
                title=self.started.strftime("%Y-%m-%d %H:%M"),
                status=status or ('<span class="live">● ライブ（2秒ごと自動更新）</span>'
                                  if live else "終了"),
                speaker_panel=speaker_panel,
                profile_panel=profile_panel,
                stats_panel=stats_panel,
                topics_panel=topics_panel,
                agent_panel=agent_panel,
                body="\n".join(parts) or '<p class="meta">（まだ発話なし）</p>',
            )
            dst = path or self.html_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(doc)
            os.replace(tmp, dst)

    def write_turns(self, recs=None, path=None):
        """discussion-support(das)のUtteranceスキーマでJSONL出力."""
        with self.state_lock:
            rs = self.records if recs is None else recs
            lines = []
            tid = 0
            for r in rs:
                if "speaker" not in r or not r.get("text"):
                    continue
                tid += 1
                lines.append(json.dumps({"turn_id": tid, "speaker": self.disp_name(r["speaker"]),
                                         "text": r["text"], "ms": r.get("ms"),
                                         "end_ms": r.get("end_ms")},
                                        ensure_ascii=False))
            dst = path or self.turns_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            os.replace(tmp, dst)

    def save(self, live: bool = True):
        self.write_md()
        self.write_html(live)
        self.write_turns()
