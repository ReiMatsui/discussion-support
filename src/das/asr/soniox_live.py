"""1マイク + Soniox リアルタイム議事録ツール（日本語・話者分離内蔵）.

【das統合版】speaker-attribution リポジトリ v1.0 からの移植（上流は凍結、以後はこちらが正）。
das連携フック: モジュール変数 ON_UTTERANCE に callable(speaker:str, text:str) を設定すると、
確定発話ごとに呼ばれる（das listen-soniox がオーケストレータへ流すのに使用）。

1本のマイクで「誰が・何を」をライブ取得する本線ツール。
Sonioxのストリーミング(WebSocket)に音声を流し、speaker付きトークンが返るので、
多マイク・ゲート・同期なしで who-said-what が出る。

機能:
  - 話者ごとに色分けしたライブ表示（確定前のテキストは薄く表示）
  - Markdown議事録 + HTML を transcripts/ に自動保存（発話確定ごと＝クラッシュ安全）
  - HTMLはブラウザ自動オープン、ライブ中2秒ごと自動更新（--no-openで無効）
  - 声紋プロファイル方式の話者特定（登録不要で自動補正）。判定は2経路のみ:
      ① 即時判定: 声紋が強一致した発話はその場で人物確定（入れ替わりも補正）
      ② それ以外は3発話バッファ: 一貫した3発話を束ねて「既存人物に合流 or 新規人物N」
      しきい値は2層: モデル別既定値 → 人物別しきい値(本人の一致sim中央値-0.12、
      新声の巻き取り防止。厳しくする方向にのみ働く)。
      不変条件: 一度確定した人物キーは書き換えない（遡及置換は 話者N→人物N の昇格のみ）。
      「1=松井」で実名化、実名のみ voices.json に永続化 → 次回から自動で実名表示。
  - 終了時に清書: 録音全体を非同期APIで再処理し、全文脈の話者分離＋声紋実名対応の
    最終版(日時.final.md/.html)を自動生成（高速応酬でのRT分離崩れへの対策。--no-polishで無効）
  - 「fix 2=1」「fix 人物2=人物1」で誤った話者の統合（過去の発言も修正）
  - 診断ログ(日時.diag.jsonl): 発話ごとの判定根拠を常時記録（問題解析用）

準備(Mac):
  uv add websockets sounddevice
  export SONIOX_API_KEY=...   # https://console.soniox.com で取得

使い方:
  uv run python offshelf/live_soniox.py            # 実マイクでライブ
  uv run python offshelf/live_soniox.py --wav offshelf/ami_raw/mic0.wav  # ファイル擬似ライブ
  実行中: 「1=松井」Enter で話者登録 / Ctrl+C で終了（保存先を表示）
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime
import json
import os
import queue
import re
import sys
import threading
import time
import unicodedata
from difflib import SequenceMatcher

import numpy as np

ON_UTTERANCE = None   # das連携: 確定発話ごとに (話者表示名, テキスト) で呼ばれる
_SYS_HOOK = None      # main()実行中のみ登録される(add_sys+saveへの橋)


def post_system(text: str) -> None:
    """das連携: ライブ議事録のタイムラインにシステム行(💡介入など)を外部から追加する."""
    if _SYS_HOOK is not None:
        _SYS_HOOK(text)

SR = 16000
WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SM_WS_URL = "wss://eu.rt.speechmatics.com/v2/"


def sm_to_res(msg: dict, lang: str = "ja") -> dict:
    """SpeechmaticsのRTメッセージをSoniox互換のトークン列に翻訳する.

    供給源を差し替えるだけで、声紋層・表示・保存・清書は無変更で動く。
    話者ラベル: S1→"1"(表示は話者1)、不明UUはそのまま。
    """
    m = msg.get("message")
    if m == "Error":
        return {"error_code": msg.get("type"), "error_message": msg.get("reason")}
    if m == "EndOfTranscript":
        return {"finished": True, "tokens": []}
    if m == "EndOfUtterance":
        return {"tokens": [{"text": "<end>", "is_final": True}]}
    if m in ("AddTranscript", "AddPartialTranscript"):
        final = m == "AddTranscript"
        toks = []
        for r in msg.get("results", []):
            alts = r.get("alternatives") or []
            content = alts[0].get("content", "") if alts else ""
            if not content:
                continue
            spk = (alts[0].get("speaker") or "UU")
            if spk.startswith("S") and spk[1:].isdigit():
                spk = spk[1:]
            if (lang not in ("ja", "zh", "cmn", "yue") and toks
                    and r.get("type") == "word"):
                content = " " + content   # 分かち書き言語は語間スペースを補う
            toks.append({"text": content, "speaker": spk,
                         "start_ms": int(r["start_time"] * 1000),
                         "end_ms": int(r["end_time"] * 1000),
                         "is_final": final})
        return {"tokens": toks}
    return {"tokens": []}   # RecognitionStarted / AudioAdded / Info / Warning 等は無視

RESET = "\x1b[0m"
DIM = "\x1b[2m"
CLEAR_LINE = "\r\x1b[K"
PALETTE = ["\x1b[36m", "\x1b[33m", "\x1b[35m", "\x1b[32m", "\x1b[34m", "\x1b[91m"]
HTML_PALETTE = ["#0e7490", "#a16207", "#7e22ce", "#15803d", "#1d4ed8", "#dc2626"]

HTML_TMPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
{refresh}<title>議事録 {title}</title>
<style>
body {{ font-family: -apple-system, "Hiragino Sans", sans-serif;
       margin: 2rem auto; padding: 0 1rem; background: #fafafa; color: #1f2937;
       max-width: 960px; }}
h1 {{ font-size: 1.2rem; }} .meta {{ color: #6b7280; font-size: .85rem; }}
.container {{ display: flex; gap: 1.2rem; align-items: flex-start; }}
.main {{ flex: 1; min-width: 0; }}
.sidebar-wrap {{ width: 200px; flex-shrink: 0; position: sticky; top: 1rem; }}
.sidebar {{ }}
.sidebar-title {{ font-size: .8rem; color: #9ca3af; margin: 0 0 .4rem; font-weight: 400; }}
.u {{ margin: .5rem 0; padding: .55rem .8rem; background: #fff; border-radius: 10px;
     border: 1px solid #e5e7eb; }}
.u .who {{ font-weight: 700; margin-right: .5em; }}
.u .ts {{ color: #9ca3af; font-size: .8rem; margin-right: .6em; font-variant-numeric: tabular-nums; }}
.live {{ color: #16a34a; font-size: .8rem; }}
.badge {{ background: #fef3c7; color: #92400e; font-size: .7rem; border-radius: 6px;
         padding: .05em .45em; margin-left: .55em; vertical-align: middle; }}
.sys {{ text-align: center; color: #6b7280; font-size: .78rem; margin: .45rem 0; }}
.speaker-panel {{ display: flex; flex-direction: column; gap: .5rem; }}
.speaker-tag {{ font-size: .85rem; padding: .4em .6em;
               background: #fff; border-radius: 8px; border: 1px solid #e5e7eb; }}
.speaker-name {{ display: flex; align-items: center; gap: .4em; font-weight: 600; }}
.speaker-name .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.rename-row {{ display: flex; gap: .25em; margin-top: .3em; }}
.rename-input {{ flex: 1; min-width: 0; font-size: .8rem; border: 1px solid #d1d5db;
                border-radius: 4px; padding: .2em .35em; }}
.rename-btn {{ font-size: .75rem; background: #2563eb; color: #fff; border: none;
              border-radius: 4px; padding: .2em .5em; cursor: pointer; white-space: nowrap; }}
.rename-btn:hover {{ background: #1d4ed8; }}
.profile-section {{ margin-bottom: .8rem; }}
.profile-item {{ display: flex; align-items: center; gap: .4em; font-size: .82rem;
                padding: .3em .5em; background: #fff; border-radius: 6px;
                border: 1px solid #e5e7eb; margin-bottom: .3em; cursor: pointer;
                user-select: none; transition: background .15s; }}
.profile-item:hover {{ background: #f3f4f6; }}
.profile-item.active {{ background: #eff6ff; border-color: #93c5fd; }}
.profile-toggle {{ width: 14px; height: 14px; border-radius: 50%;
                  border: 2px solid #d1d5db; flex-shrink: 0; transition: all .15s; }}
.profile-item.active .profile-toggle {{ background: #2563eb; border-color: #2563eb; }}
.stats-section {{ margin-bottom: .8rem; }}
.stats-group {{ margin-bottom: .6rem; }}
.stats-label {{ font-size: .7rem; color: #9ca3af; margin-bottom: .2rem; }}
.stats-row {{ display: flex; align-items: center; gap: .4em; font-size: .78rem; margin-bottom: .2em; }}
.stats-name {{ width: 3.5em; flex-shrink: 0; text-align: right; overflow: hidden;
              text-overflow: ellipsis; white-space: nowrap; }}
.stats-bar-bg {{ flex: 1; height: 10px; background: #e5e7eb; border-radius: 5px; overflow: hidden; }}
.stats-bar {{ height: 100%; border-radius: 5px; transition: width .3s; }}
.stats-pct {{ width: 2.8em; flex-shrink: 0; font-size: .72rem; color: #6b7280; font-variant-numeric: tabular-nums; }}
.topics-section {{ margin-top: .4rem; }}
.topic-item {{ font-size: .78rem; padding: .3em .5em; margin-bottom: .25em;
              background: #fff; border-radius: 6px; border: 1px solid #e5e7eb;
              border-left: 3px solid #8b5cf6; }}
.topic-text {{ color: #1f2937; }}
.topic-by {{ font-size: .68rem; color: #9ca3af; }}
.agent-section {{ margin-top: .6rem; padding: .5em; background: #f0f9ff; border-radius: 8px;
                 border: 1px solid #bae6fd; }}
.agent-header {{ display: flex; align-items: center; justify-content: space-between; }}
.agent-label {{ font-size: .82rem; font-weight: 600; color: #0369a1; }}
.agent-conn {{ font-size: .68rem; color: #6b7280; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }}
.agent-modes {{ display: flex; gap: .25em; margin-top: .35em; }}
.agent-mode-btn {{ font-size: .72rem; padding: .2em .5em; border: 1px solid #93c5fd;
                  border-radius: 5px; background: #fff; color: #1e40af; cursor: pointer;
                  transition: all .15s; flex: 1; text-align: center; }}
.agent-mode-btn:hover {{ background: #dbeafe; }}
.agent-mode-btn.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
.agent-opts {{ display: flex; gap: .5em; margin-top: .35em; align-items: center; }}
.agent-opt-label {{ font-size: .7rem; color: #6b7280; display: flex; align-items: center; gap: .25em; }}
.agent-select {{ font-size: .7rem; border: 1px solid #d1d5db; border-radius: 4px;
                padding: .15em .25em; background: #fff; }}
.agent-num {{ width: 2.5em; font-size: .7rem; border: 1px solid #d1d5db; border-radius: 4px;
             padding: .15em .25em; text-align: center; }}
.agent-trigger-row {{ display: none; }}
.agent-section[data-mode="facilitator"] .agent-trigger-row {{ display: flex; }}
</style></head><body>
<h1>議事録 {title}</h1>
<p class="meta">{status}</p>
<div class="container">
<div class="main">
{body}
</div>
<div class="sidebar-wrap">
{profile_panel}
{speaker_panel}
{stats_panel}
{topics_panel}
{agent_panel}
</div>
</div>
<script>
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
async function _agentCfg(data) {{
  try {{
    var res = await fetch('/agent', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(data)
    }});
    if (res.ok) {{ var d = await res.json(); _updateAgentUI(d); }}
  }} catch(e) {{}}
}}
function _updateAgentUI(d) {{
  var sec = document.querySelector('.agent-section');
  if (!sec) return;
  sec.dataset.mode = d.mode || 'off';
  sec.querySelectorAll('.agent-mode-btn').forEach(function(b) {{
    b.classList.toggle('active', b.dataset.mode === d.mode);
  }});
}}
function setAgentMode(btn) {{
  _agentCfg({{mode: btn.dataset.mode}});
}}
function setAgentVoice(sel) {{
  _agentCfg({{voice: sel.value}});
}}
function setAgentTrigger(inp) {{
  var v = parseInt(inp.value);
  if (v > 0) _agentCfg({{trigger_n: v}});
}}
async function toggleProfile(el) {{
  var name = el.dataset.name;
  var isActive = el.classList.contains('active');
  try {{
    var res = await fetch('/activate', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{name: name, active: !isActive}})
    }});
    if (res.ok) location.reload();
    else {{ var d = await res.json(); alert(d.error || '切替失敗'); }}
  }} catch(e) {{}}
}}
async function rename(btn) {{
  var input = btn.parentElement.querySelector('.rename-input');
  var label = input.dataset.label;
  var name = input.value.trim();
  if (!name) return;
  btn.disabled = true;
  try {{
    var res = await fetch('/rename', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{label: label, name: name}})
    }});
    if (res.ok) location.reload();
    else {{ var d = await res.json(); alert(d.error || '登録失敗'); btn.disabled = false; }}
  }} catch(e) {{ btn.disabled = false; }}
}}
document.querySelectorAll('.rename-input').forEach(function(input) {{
  input.addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') rename(input.nextElementSibling);
  }});
}});
window.scrollTo(0, document.body.scrollHeight);
</script>
</body></html>
"""


def load_env(path: str = ".env") -> None:
    """プロジェクト直下の .env からAPIキー等を読み込む（既に設定済みの環境変数を優先）.

    形式: KEY=VALUE の行（#始まりはコメント）。依存なしの最小実装。
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def fmt_ts(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------- 論点抽出（非同期LLM処理） ----------

OPENAI_API = "https://api.openai.com/v1/chat/completions"

_TOPIC_PROMPT = """\
あなたは会議の論点を抽出するアシスタントです。

## 既存の論点
{existing}

## 直近の発話
{utterances}

## 指示
直近の発話の中に、既存の論点リストに**まだ含まれていない新しい論点**があれば抽出してください。
各論点について、最初にその論点を提起した発話者名を特定してください。

出力はJSON配列のみ（説明不要）。新しい論点がなければ空配列 [] を返してください。
形式: [{{"topic": "論点の短い要約", "speaker": "発話者名"}}]"""


def _extract_topics(utterances: list[dict], existing: list[str],
                    api_key: str, model: str) -> list[dict]:
    """OpenAI APIで新論点を抽出する（同期呼び出し、バックグラウンドスレッド用）."""
    if not utterances or not api_key:
        return []
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    ex_text = "\n".join(f"- {t}" for t in existing) if existing else "（まだなし）"
    prompt = _TOPIC_PROMPT.format(existing=ex_text, utterances=utt_text)
    # GPT-5系/o系はtemperature指定不可、max_tokensはmax_completion_tokensに改名
    name = model.lower()
    is_new = name.startswith(("gpt-5", "o1", "o3", "o4"))
    params: dict = {"model": model,
                    "messages": [{"role": "user", "content": prompt}]}
    if not is_new:
        params["temperature"] = 0.3
        params["max_tokens"] = 512
    else:
        params["max_completion_tokens"] = 512
    body = json.dumps(params).encode()
    import urllib.request
    req = urllib.request.Request(OPENAI_API, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        text = resp["choices"][0]["message"]["content"].strip()
        # JSON配列を抽出（前後にmarkdownコードブロックがある場合も対応）
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception:
        return []


# ---------- AIエージェント（Realtime API v2） ----------

_PROMPT_FACILITATOR = """\
あなたは会議のファシリテーターAIです。
参加者の議論を聞いて、必要な時だけ介入してください。

介入すべき場面:
- 議論が行き詰まった時（新しい視点を提案）
- 重要な論点が見落とされている時
- 議論が脱線した時（元のテーマに戻す提案）
- 合意形成が必要な時（要約して確認）

不必要に発言しないでください。人間の議論を尊重し、
本当に価値ある貢献ができる時だけ簡潔に発言してください。
発言は日本語で、30秒以内に収まる長さにしてください。

もし介入が不要だと判断した場合は、「（介入不要）」とだけ返してください。"""

_PROMPT_CONVERSATION = """\
あなたは会議に参加しているAIアシスタントです。
参加者と自然に会話してください。質問されたら必ず答えてください。
簡潔に、日本語で返答してください（15秒以内に収まる長さ）。
会議の文脈を踏まえた上で、役に立つ回答を心がけてください。"""

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"
AGENT_SPEAKER = "AI"          # recordsに使うスピーカーキー
_AGENT_TRIGGER = 10           # N発話ごとに応答検討(facilitator)
_AGENT_SILENCE = 5.0          # N秒沈黙で応答検討
AGENT_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]


# ---------- 信号レベルAEC（エコーキャンセレーション） ----------


def _resample_24_to_16(pcm_24k: np.ndarray) -> np.ndarray:
    """24kHz float32 → 16kHz float32 リサンプル（線形補間）。AEC参照信号用。"""
    n_in = len(pcm_24k)
    n_out = int(n_in * 2 / 3)
    if n_out < 2:
        return np.empty(0, dtype=np.float32)
    idx = np.linspace(0, n_in - 1, n_out)
    return np.interp(idx, np.arange(n_in), pcm_24k).astype(np.float32)


class _EchoCanceller:
    """信号レベルAEC: AI再生音声（参照信号）をマイク入力から減算。

    AI音声のPCMデータを完全に保持しているので、FFT相互相関で遅延を推定し、
    ゲインを合わせて減算する。人間の声は参照に含まれないのでそのまま通る。
    電話やビデオ会議ソフトと同じ原理。
    """

    def __init__(self, sr: int = 16000, max_delay_s: float = 0.5):
        self.sr = sr
        self.max_delay = int(max_delay_s * sr)   # 最大探索遅延（サンプル）
        # リングバッファ: 参照信号を10秒分保持
        self._buf_len = sr * 10
        self._ref = np.zeros(self._buf_len, dtype=np.float32)
        self._wpos = 0
        self._lock = threading.Lock()
        # 推定パラメータ
        self._delay = 0
        self._gain = 0.0
        self._estimated = False
        self._n_frames = 0
        self._re_est_every = 20     # N フレームごとに再推定

    def feed_reference(self, pcm_16k: np.ndarray):
        """16kHzリサンプル済み参照音声をバッファに追加。再生スレッドから呼ぶ。"""
        with self._lock:
            n = len(pcm_16k)
            if n == 0:
                return
            L = self._buf_len
            w = self._wpos
            end = w + n
            if end <= L:
                self._ref[w:end] = pcm_16k
            else:
                first = L - w
                self._ref[w:] = pcm_16k[:first]
                self._ref[:n - first] = pcm_16k[first:]
            self._wpos = end % L

    def _get_ref_unlocked(self, length: int) -> np.ndarray:
        """直近 length サンプルを取得。_lock 保持下で呼ぶこと。"""
        L = self._buf_len
        w = self._wpos
        length = min(length, L)
        start = (w - length) % L
        if start + length <= L:
            return self._ref[start:start + length].copy()
        first = L - start
        return np.concatenate([self._ref[start:], self._ref[:length - first]])

    def process(self, mic: np.ndarray) -> np.ndarray:
        """マイクフレームからAIエコーを除去して返す。senderスレッドから呼ぶ。"""
        flen = len(mic)
        need = flen + self.max_delay

        with self._lock:
            ref_seg = self._get_ref_unlocked(need)

        # 参照にエネルギーがなければスキップ（AI無音中）
        if np.mean(ref_seg ** 2) < 1e-8:
            self._estimated = False
            return mic

        self._n_frames += 1

        # 定期的に遅延・ゲインを再推定
        if not self._estimated or self._n_frames % self._re_est_every == 0:
            self._estimate(mic, ref_seg)

        if not self._estimated:
            return mic

        # 推定遅延で参照を切り出し
        # ref_seg[k : k+flen] が delay = max_delay - k に対応
        k = self.max_delay - self._delay
        if k < 0 or k + flen > len(ref_seg):
            return mic
        ref_aligned = ref_seg[k: k + flen]

        # ゲイン適応（エコーの相対音量）
        rp = np.dot(ref_aligned, ref_aligned)
        if rp > 1e-8:
            g = float(np.clip(np.dot(mic, ref_aligned) / rp, 0.0, 3.0))
            self._gain = 0.7 * self._gain + 0.3 * g

        # 減算
        return mic - self._gain * ref_aligned

    def _estimate(self, mic: np.ndarray, ref_seg: np.ndarray):
        """FFT相互相関で遅延を推定。"""
        flen = len(mic)
        N = 1
        while N < len(ref_seg) + flen:
            N *= 2
        MIC = np.fft.rfft(mic, N)
        REF = np.fft.rfft(ref_seg, N)
        # xcorr[k] = sum_i mic[i] * ref_seg[i+k]
        xcorr = np.fft.irfft(np.conj(MIC) * REF, N)
        search = np.abs(xcorr[: self.max_delay + 1])
        best_k = int(np.argmax(search))
        # 相関が弱すぎる場合はスキップ
        mic_energy = np.dot(mic, mic)
        if mic_energy < 1e-8:
            return
        ncc = search[best_k] / (np.sqrt(mic_energy * np.dot(ref_seg, ref_seg)) + 1e-8)
        if ncc < 0.01:
            return
        self._delay = self.max_delay - best_k
        self._estimated = True


class RealtimeAgent:
    """OpenAI Realtime API v2 WebSocket で会議に参加するAIエージェント.

    エコー防止（マイク常時オン — 人間の割り込みを維持）:
      1. 信号レベルAEC — AI再生音声を参照信号としてマイク入力から減算（sender）
      2. flush() 2層テキスト判定（議事録フィルタ）:
         a. 既知AIスピーカー + 類似度>0.25 → 除去
         b. 強テキスト一致>0.40 → 除去（クールダウン中ならスピーカー学習）
      3. クールダウン — AI発話中〜終了後5秒はカーソルを止めてバックログ。
         flush()でエコー除去済みのrecordsをクールダウン後にまとめて処理。
      4. _agent_worker テキストフィルタ — 2重フィルタとしてflush()通過後に再チェック
      5. 応答状態ガード — 応答生成中は新規triggerを抑止

    モード:
      off          = 無効
      facilitator  = N発話 or 沈黙でトリガー、介入不要なら黙る
      conversation = 毎発話でトリガー、必ず返答する
    """

    MODES = ("off", "facilitator", "conversation")

    def __init__(self, api_key: str, voice: str = "alloy",
                 mode: str = "facilitator", trigger_n: int = _AGENT_TRIGGER):
        self.api_key = api_key
        self.voice = voice
        self.mode = mode                   # off / facilitator / conversation
        self.trigger_n = trigger_n
        self.ws = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending: list[dict] = []     # 送信待ち発話
        self.ai_speaking = False           # AI音声再生中フラグ
        self._ai_text_buf = ""             # ストリーミング転写バッファ
        self._audio_q: "queue.Queue[bytes | None]" = queue.Queue()  # ストリーミング再生用
        self._connected = False
        self._conn_error = ""              # 接続エラーメッセージ（UI表示用）
        self.on_ai_utterance = None        # callback(text: str) AI発話確定時
        self._playback_thread: threading.Thread | None = None
        # --- エコー防止 ---
        self._responding = False           # response生成中フラグ
        self._echo_canceller = _EchoCanceller()
        self._recent_ai_texts: collections.deque = collections.deque(maxlen=20)
        self._last_speech_end = 0.0        # ai_speaking が False になった時刻
        self._echo_cooldown = 5.0          # AI発話終了後のクールダウン秒数
        # スピーカーID自動検出: エコーと判定されたvoiceprint sp_idを記録
        self._ai_echo_speakers: dict[str, float] = {}  # sp_id -> 最終検出時刻(monotonic)

    @property
    def _prompt(self) -> str:
        return _PROMPT_CONVERSATION if self.mode == "conversation" else _PROMPT_FACILITATOR

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def connect(self):
        """WebSocket接続を開始し、受信スレッドを起動."""
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._conn_error = "websockets未インストール"
            print("# AI Agent: websockets がインストールされていません", flush=True)
            return
        try:
            self.ws = connect(
                REALTIME_URL,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
        except Exception as e:
            self._conn_error = str(e)[:80]
            print(f"# AI Agent: 接続失敗 ({e})", flush=True)
            return
        self._connected = True
        self._conn_error = ""
        self._send_session_update()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self._start_playback_thread()
        print(f"# AI Agent: 接続完了（voice={self.voice}, mode={self.mode}）", flush=True)

    def _send_session_update(self):
        """現在の設定でsession.updateを送信（GA API形式）.

        GA (gpt-realtime-2) WebSocket スキーマ:
          session.type = "realtime"           (必須)
          session.instructions               (フラット)
          session.audio.input.turn_detection  (None で VAD 無効)
          session.audio.output.voice          (ネスト)
        参照: https://developers.openai.com/api/docs/guides/realtime-conversations
        """
        if not self.ws:
            return
        try:
            self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": self._prompt,
                    "audio": {
                        "input": {
                            "turn_detection": None,
                        },
                        "output": {
                            "voice": self.voice,
                        },
                    },
                },
            }))
        except Exception as e:
            print(f"# AI Agent: session.update失敗 ({e})", flush=True)

    def apply_config(self, mode: str | None = None, voice: str | None = None,
                     trigger_n: int | None = None):
        """動的に設定変更（UIから呼ばれる）."""
        changed = False
        if mode is not None and mode in self.MODES and mode != self.mode:
            self.mode = mode
            changed = True
        if voice is not None and voice in AGENT_VOICES and voice != self.voice:
            self.voice = voice
            changed = True
        if trigger_n is not None and trigger_n > 0:
            self.trigger_n = trigger_n
        if changed and self._connected:
            self._send_session_update()

    # --- ストリーミング音声再生 ---

    def _start_playback_thread(self):
        """PCMキューから読み出して逐次再生するスレッド。AEC参照信号も同時にバッファ。"""
        def _player():
            try:
                import sounddevice as sd
                stream = sd.OutputStream(samplerate=24000, channels=1,
                                         dtype="float32", blocksize=2400)
                stream.start()
                while not self._stop.is_set():
                    chunk = self._audio_q.get()
                    if chunk is None:          # 1応答の終端
                        self.ai_speaking = False
                        self._last_speech_end = time.monotonic()
                        continue
                    pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                    stream.write(pcm.reshape(-1, 1))
                    # AEC: 再生音声を16kHzにリサンプルして参照バッファに蓄積
                    ref16 = _resample_24_to_16(pcm)
                    if len(ref16) > 0:
                        self._echo_canceller.feed_reference(ref16)
                stream.stop()
                stream.close()
            except Exception as e:
                print(f"# AI音声再生スレッド異常: {e}", flush=True)

        self._playback_thread = threading.Thread(target=_player, daemon=True)
        self._playback_thread.start()

    # --- WebSocket受信 ---

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw = self.ws.recv()
                ev = json.loads(raw)
            except Exception as e:
                if not self._stop.is_set():
                    self._conn_error = f"切断: {e}"[:80]
                    print(f"# AI Agent: WebSocket切断 ({e})", flush=True)
                break
            self._handle(ev)
        self._connected = False

    def _handle(self, ev: dict):
        etype = ev.get("type", "")

        if etype == "response.output_audio.delta":
            chunk = ev.get("delta", "")
            if chunk:
                self._audio_q.put(base64.b64decode(chunk))
                self.ai_speaking = True

        elif etype == "response.output_audio_transcript.delta":
            self._ai_text_buf += ev.get("delta", "")

        elif etype == "response.output_audio_transcript.done":
            transcript = ev.get("transcript", "") or self._ai_text_buf
            self._ai_text_buf = ""
            if transcript and "（介入不要）" not in transcript:
                self._recent_ai_texts.append(transcript)
                if self.on_ai_utterance:
                    self.on_ai_utterance(transcript)

        elif etype == "response.output_audio.done":
            self._audio_q.put(None)   # 再生終端マーカー

        elif etype == "response.done":
            self._ai_text_buf = ""
            self._responding = False

        elif etype == "error":
            msg = ev.get("error", {}).get("message", "unknown")
            print(f"# AI Agent エラー: {msg}", flush=True)

    # --- 発話送信 ---

    def feed(self, speaker: str, text: str):
        """人間の確定発話をエージェントに蓄積."""
        if not self._connected or not self.enabled:
            return
        with self._lock:
            self._pending.append({"speaker": speaker, "text": text})

    def trigger(self):
        """蓄積した発話をRealtimeAPIに送信し応答を要求."""
        if not self._connected or not self.enabled or not self.ws:
            return
        if self._responding:
            return  # 応答生成中は新規リクエストを抑止
        with self._lock:
            if not self._pending:
                return
            conv = "\n".join(f"{u['speaker']}: {u['text']}" for u in self._pending)
            self._pending.clear()
        try:
            self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": conv}],
                },
            }))
            self.ws.send(json.dumps({"type": "response.create"}))
            self._responding = True
        except Exception as e:
            print(f"# AI Agent 送信エラー: {e}", flush=True)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def in_echo_cooldown(self) -> bool:
        """AI発話中、またはAI発話終了後のクールダウン期間中か。"""
        if self.ai_speaking or self._responding:
            return True
        return time.monotonic() - self._last_speech_end < self._echo_cooldown

    def cancel_echo(self, mic_float: np.ndarray) -> np.ndarray:
        """マイク入力(float32, 16kHz)からAIエコーを除去。senderスレッドから呼ぶ。"""
        return self._echo_canceller.process(mic_float)

    @staticmethod
    def _normalize(text: str) -> str:
        """テキスト比較用の正規化: 句読点・空白・記号を除去。"""
        t = unicodedata.normalize("NFKC", text)
        return re.sub(r'[\s　、。,.!?！？「」『』（）()・…\-―ー～~]+', '', t)

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> set[str]:
        """文字n-gramの集合を返す。"""
        if len(text) < n:
            return {text} if text else set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def _best_similarity(self, text: str) -> float:
        """正規化テキストとAI生成テキスト群の最大類似度を返す（0.0〜1.0）。"""
        if not text or not self._recent_ai_texts:
            return 0.0
        norm = self._normalize(text)
        if len(norm) < 2:
            return 0.0
        best = 0.0
        for ai_text in self._recent_ai_texts:
            ai_norm = self._normalize(ai_text)
            # 部分一致: 完全包含なら1.0
            if len(norm) >= 4 and norm in ai_norm:
                return 1.0
            if len(ai_norm) >= 4 and ai_norm in norm:
                return 1.0
            # SequenceMatcher
            sm = SequenceMatcher(None, norm, ai_norm).ratio()
            # 文字trigram Jaccard類似度
            ng_a = self._char_ngrams(norm)
            ng_b = self._char_ngrams(ai_norm)
            jaccard = len(ng_a & ng_b) / max(len(ng_a | ng_b), 1)
            # 両方の最大値を採用（STTの揺れに強い）
            best = max(best, sm, jaccard)
        return best

    def is_ai_echo(self, speaker_id: str) -> bool:
        """このスピーカーが過去にAIエコーとして検出されたことがあるか。
        60秒以上検出がなければ自動解除（誤マーク防止）。"""
        t = self._ai_echo_speakers.get(speaker_id)
        if t is None:
            return False
        if time.monotonic() - t > 60.0:
            del self._ai_echo_speakers[speaker_id]
            return False
        return True

    def mark_ai_echo(self, speaker_id: str):
        """スピーカーをAIエコーとして記録（タイムスタンプ付き）。"""
        self._ai_echo_speakers[speaker_id] = time.monotonic()

    def close(self):
        self._stop.set()
        self._audio_q.put(None)
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass


# ---------- 清書（会議後の非同期再処理） ----------
# RTの話者分離は速い応酬で崩れる(実測: 高速応酬区間で1ラベルに併合)。非同期APIは
# 全文脈を見られるため分離精度が大幅に高い(公式)。終了時に録音全体を再処理し、
# async話者を声紋プロファイルで実名に対応づけて「清書版」議事録を作る。

API_BASE = "https://api.soniox.com"


def _wav_bytes(pcm: bytes) -> bytes:
    import struct
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) +
            b"data" + struct.pack("<I", n) + pcm)


def _api(api_key: str, method: str, path: str, body=None, ctype=None, timeout=120):
    import urllib.request
    req = urllib.request.Request(API_BASE + path, data=body, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    if ctype:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def _group_tokens(tokens: list[dict]) -> list[tuple]:
    """async結果のトークン列を (start_ms, end_ms, 話者, テキスト) の発話列へ."""
    utts = []
    cur = None   # [start, end, spk, text]
    for tk in tokens:
        text = tk.get("text") or ""
        if not text or text == "<end>":
            continue
        spk = tk.get("speaker")
        if cur is None or spk != cur[2]:
            if cur and cur[3].strip():
                utts.append(tuple(cur))
            cur = [tk.get("start_ms"), tk.get("end_ms"), spk, ""]
        if tk.get("end_ms") is not None:
            cur[1] = tk["end_ms"]
        cur[3] += text
    if cur and cur[3].strip():
        utts.append(tuple(cur))
    return utts


def _map_speakers(utts: list[tuple], pcm: bytes, tracker) -> dict:
    """async話者ID → 表示キー（人物との1対1割当）.

    各async話者の長い発話の声紋平均をプロファイルと照合し、類似の高いペアから
    貪欲に1対1で割り当てる。1対1にしないと、同一再生チェーン等で複数のasync話者が
    同じ人物に畳まれ、清書の話者数がライブより減る事故が起きる（2026-06-12実測）。
    """
    mapping = {}
    if tracker is None:
        return mapping
    by_spk: dict = {}
    for s, e, spk, _ in utts:
        if s is None or e is None or spk is None:
            continue
        by_spk.setdefault(str(spk), []).append((e - s, s, e))
    # アクティブなプロファイルのみ対象（セッション中に使ったもの＋自動登録）
    active = {k: v for k, v in tracker.profiles.items() if k in tracker._active_keys}
    pairs = []   # (sim, async話者, 人物)
    for spk, segs in by_spk.items():
        segs = [x for x in sorted(segs, reverse=True) if x[0] >= 1200][:6]
        embs = []
        for _, s, e in segs:
            wav = np.frombuffer(pcm[s * 32: e * 32], dtype="<i2").astype(np.float32) / 32768.0
            emb = tracker._embed(wav)
            if emb is not None:
                embs.append(emb)
        if embs:
            prof = np.mean(embs, axis=0)
            prof = prof / np.linalg.norm(prof)
            for n, v in active.items():
                sim = float(np.dot(v, prof))
                if sim >= tracker.dedupe:
                    pairs.append((sim, spk, n))
    used_spk, used_person = set(), set()
    for sim, spk, n in sorted(pairs, reverse=True):
        if spk in used_spk or n in used_person:
            continue
        mapping[spk] = n
        used_spk.add(spk)
        used_person.add(n)
    return mapping


def polish(api_key: str, pcm: bytes, lang: str, tracker, log=print) -> list[dict]:
    """録音全体を非同期APIで再処理し、清書版のrecordsを返す."""
    log("# 清書: 音声をアップロード中…")
    import uuid
    b = "----spkattr" + uuid.uuid4().hex
    body = ((f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"meeting.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
            + _wav_bytes(pcm) + f"\r\n--{b}--\r\n".encode())
    file_id = _api(api_key, "POST", "/v1/files", body,
                   f"multipart/form-data; boundary={b}", timeout=600)["id"]
    tid = None
    try:
        cfg = {"model": "stt-async-v4", "language_hints": [lang],
               "enable_speaker_diarization": True, "file_id": file_id}
        tid = _api(api_key, "POST", "/v1/transcriptions",
                   json.dumps(cfg).encode(), "application/json")["id"]
        log("# 清書: 再処理を待っています…")
        t0 = time.time()
        while True:
            st = _api(api_key, "GET", f"/v1/transcriptions/{tid}")
            if st["status"] == "completed":
                break
            if st["status"] == "error":
                raise RuntimeError(st.get("error_message", "unknown"))
            if time.time() - t0 > 600:
                raise TimeoutError("非同期処理が10分以内に完了しませんでした")
            time.sleep(2)
        tokens = _api(api_key, "GET", f"/v1/transcriptions/{tid}/transcript")["tokens"]
    finally:   # 後始末（失敗しても続行）
        try:
            if tid:
                _api(api_key, "DELETE", f"/v1/transcriptions/{tid}")
            _api(api_key, "DELETE", f"/v1/files/{file_id}")
        except Exception:
            pass
    utts = _group_tokens(tokens)
    log(f"# 清書: {len(utts)}発話を取得、話者を声紋で照合中…")
    mapping = _map_speakers(utts, pcm, tracker)
    return [{"ms": s, "speaker": mapping.get(str(spk), "#" + str(spk)), "text": tx.strip()}
            for s, e, spk, tx in utts]


class VoiceProfiles:
    """凍結プロファイル照合による話者特定（台帳固定・誤り非伝播）.

    判定は2経路だけ:
      ① 即時判定 — 単発声紋が強一致(thresh＋2位とmargin差)した時だけ、その場で人物確定
      ② それ以外は3発話バッファ — 一貫した3発話を束ね「既存人物に合流(dedupe) or 新規人物N」
    しきい値は2層構造（厳しくする方向にのみ働き、最悪でも既定値の挙動に戻る）:
      1. モデル別既定値(DEFAULTS)
      2. 人物別しきい値(その人物の一致sim中央値-0.12 = 新規性検出。中途半端な類似の
         新しい声を既存人物に巻き取らない)。即時判定のみに適用
    不変条件: 確定済みの人物キーは書き換えない（遡及置換は #ラベル→人物 の昇格のみ）。
    実名(enroll)のみ voices.json に永続化、匿名「人物N」はセッション限り。
    """

    ANON = re.compile(r"^人物\d+$")

    # モデル別の既定しきい値（実音声プールで校正済み。スコアのスケールが違う）
    # resemblyzer: 軽量・依存少。同一/別人の分布に重なりあり（分離マージン-0.06）
    # ecapa: ほぼ完全分離(+0.01)＋10倍速。混合音声を成分話者と強くマッチさせる癖
    # redimnet: Interspeech 2024。本プールで最良の分離(+0.10)・27ms級・5M params
    # (即時判定th, 合流dedupe, 一貫性consist)。dedupeは三発話プロファイル同士の比較なので
    # 単発より高め（2026-06-11夜: 0.30→巻き取り復活/個人別→本人分裂のため固定の中庸値に）
    DEFAULTS = {"resemblyzer": (0.75, 0.72, 0.62), "ecapa": (0.35, 0.40, 0.30),
                "redimnet": (0.42, 0.50, 0.34)}

    def __init__(self, path: str = "voices.json", thresh: float | None = None,
                 min_sec: float = 1.0, margin: float = 0.05, auto: bool = True,
                 consist: float | None = None, dedupe: float | None = None,
                 model: str = "resemblyzer"):
        self.model = model
        if model == "ecapa":
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

            def _embed_raw(wav):
                with torch.no_grad():
                    return enc.encode_batch(torch.from_numpy(wav).float().unsqueeze(0)).squeeze().numpy()
            self._embed_raw = _embed_raw
        elif model == "redimnet":
            import torch   # 初回はGitHubからコード＋重み(20MB)をダウンロード
            enc = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="b2",
                                 train_type="ft_lm", dataset="vox2", trust_repo=True)
            enc.eval()

            def _embed_raw(wav):
                with torch.no_grad():
                    return enc(torch.from_numpy(wav).float().unsqueeze(0)).squeeze().numpy()
            self._embed_raw = _embed_raw
        else:
            from resemblyzer import VoiceEncoder, preprocess_wav  # 初回ロード数秒
            enc = VoiceEncoder("cpu", verbose=False)
            self._embed_raw = lambda wav: enc.embed_utterance(preprocess_wav(wav, source_sr=SR))
        d_th, d_dd, d_cs = self.DEFAULTS[model]
        self.path = path
        self.thresh = thresh if thresh is not None else d_th   # 即時判定のしきい値
        self.margin = margin   # 即時判定の追加条件: 2位との差（似た声の誤マッチ防止）
        self.auto = auto       # 未知の声の自動登録（匿名「人物N」プロファイル）
        self.consist = consist if consist is not None else d_cs  # 3発話の全ペア類似の下限
        self.dedupe = dedupe if dedupe is not None else d_dd     # 既存人物への合流しきい値
        self.min_sec = min_sec
        self.profiles: dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.pop("_model", "resemblyzer") == model:   # 別モデルの声紋は互換性なし
                self.profiles = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
            else:
                print(f"# 注意: {path} は別の声紋モデルで作成されたため読み込みません", flush=True)
        self.sp_map: dict[str, str] = {}                    # Sonioxラベル -> 表示キー
        self.label_embs: dict[str, list[np.ndarray]] = {}   # ラベル -> 直近声紋（手動登録・校正用）
        # 未確定の声のプール（ラベルで仕切らない）。Sonioxは新しい声を既存ラベルに混ぜて
        # 出すことがあり、ラベル別バッファだと他話者の混入で3発話一貫が永遠に成立しない
        # （実セッション診断: 蓄積33回vs登録3回）。声は声同士で束ねる。
        self.pool: list[np.ndarray] = []
        self.n_anon = 0
        # 部屋の分布計測（表示・診断専用、判定には使わない）。かつてしきい値の自動校正に
        # 使っていたが、実セッションで未発動＋「ラベル=人物」前提が崩れている(Sonioxは
        # 新しい声を既存ラベルに混ぜる)＋人物別しきい値が同じ役割をより清潔なデータで
        # 果たすため、判定への結線は撤去した(2026-06-11)。
        self.same_sims: list[float] = []
        self.diff_sims: list[float] = []
        # 人物別しきい値: 「その人物が普段一致するスコアの典型範囲」を下回る一致は弾く
        # （新規性検出）。同一再生チェーン等で別人が0.5前後の中途半端な類似を出しても、
        # 本人の典型(例:0.7台)に届かなければ巻き取らない。診断ログ解析(2026-06-11)で
        # 吸収帯0.45-0.59と本人帯0.67-0.82の分離を確認、検証で吸収率91%→0%。
        self.own_sims: dict[str, list[float]] = {}   # 人物 -> 受理された一致simの履歴
        self.embed_ms: list[float] = []                     # レイテンシ統計
        self.counts: dict[str, int] = {}                    # 判定種別の集計
        self.last: dict | None = None                       # 直近の判定内容（可視化用）
        self._lock = threading.RLock()   # classify(受信スレッド)とenroll/remap(入力スレッド)の排他
        # プロファイル選択: セッション中に照合対象とするプロファイルのキー集合。
        # voices.jsonから読んだ名前付きプロファイルは全て非アクティブで開始し、
        # ユーザーが明示的にONにしたもの＋セッション中に自動登録された人物Nのみが照合対象。
        self._active_keys: set[str] = set()

    def _note(self, kind: str, **info) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.last = {"kind": kind, **info}

    def _update_room_stats(self, sp: str, emb: np.ndarray) -> None:
        for l2, es in self.label_embs.items():
            tgt = self.same_sims if l2 == sp else self.diff_sims
            tgt.extend(float(np.dot(emb, e2)) for e2 in es[-3:])
        del self.same_sims[:-60]
        del self.diff_sims[:-120]

    def _person_th(self, name: str, base: float) -> float:
        """人物別しきい値 = max(基準値, その人物の一致sim中央値 - 0.12)."""
        h = self.own_sims.get(name, [])
        if len(h) >= 3:
            return max(base, float(np.median(h)) - 0.12)
        return base

    def _embed(self, wav: np.ndarray) -> np.ndarray | None:
        t0 = time.perf_counter()
        try:
            emb = self._embed_raw(wav)
            if emb is None or np.asarray(emb).ndim != 1:
                return None
        except Exception:
            return None
        self.embed_ms.append((time.perf_counter() - t0) * 1000)
        emb = np.asarray(emb, dtype=np.float64)
        return emb / np.linalg.norm(emb)

    def classify(self, wav: np.ndarray, sp, overlapped: bool = False) -> str:
        """発話を人物キーに割り当てる（経路はクラスdocstring参照）.

        overlapped=True の発話は声が混ざっていて声紋がデタラメになるため、
        声での判定をスキップして直前の対応を維持する。
        """
        with self._lock:
            return self._classify(wav, sp, overlapped)

    def _classify(self, wav: np.ndarray, sp, overlapped: bool) -> str:
        sp = str(sp)
        prev = self.sp_map.get(sp)
        kind, info = "相槌追従", {}
        if overlapped and wav.size >= SR * self.min_sec:
            kind = "重なりスキップ"
        elif wav.size >= SR * self.min_sec:
            emb = self._embed(wav)
            if emb is None:
                kind = "声紋計算不可"
            else:
                self._update_room_stats(sp, emb)   # 部屋の同一/別人分布を実測(表示・診断用)
                self.label_embs.setdefault(sp, []).append(emb)
                del self.label_embs[sp][:-10]    # 手動登録用に直近10発話だけ保持
                th, dd, cs = self.thresh, self.dedupe, self.consist
                active = {k: v for k, v in self.profiles.items() if k in self._active_keys}
                info = {"n_prof": len(active), "n_all": len(self.profiles)}   # 診断ログ用
                if active:
                    ranked = sorted(((float(np.dot(p, emb)), n)
                                     for n, p in active.items()), reverse=True)
                    sim, cand = ranked[0]
                    second = ranked[1][0] if len(ranked) > 1 else -1.0
                    info.update(sim=round(sim, 3), second=round(second, 3), name=cand, prev=prev)
                    if sim >= self._person_th(cand, th) and sim - second >= self.margin:
                        # 注: ここでbufは消さない。たまたま強一致した発話でバッファを
                        # リセットすると、新しい話者の3発話が永遠に貯まらない(検証で確認)。
                        self.sp_map[sp] = cand
                        h = self.own_sims.setdefault(cand, [])
                        h.append(sim)
                        del h[:-20]
                        self._note("補正" if (prev is not None and not prev.startswith("#")
                                              and prev != cand) else "声紋一致", label=sp, **info)
                        return cand
                kind = "蓄積中" if self.auto else "未確定"
                if self.auto:
                    # 声プール: ラベル不問で、互いに一貫する3発話が揃ったら人物化
                    sims = sorted(((float(np.dot(p, emb)), i) for i, p in enumerate(self.pool)),
                                  reverse=True)
                    cand = [i for s, i in sims[:2] if s >= cs]
                    if len(cand) == 2 and float(np.dot(self.pool[cand[0]],
                                                       self.pool[cand[1]])) >= cs:
                        triple = [self.pool[cand[0]], self.pool[cand[1]], emb]
                        for i in sorted(cand, reverse=True):
                            self.pool.pop(i)
                        prof = np.mean(triple, axis=0)
                        prof = prof / np.linalg.norm(prof)
                        hit_sim, hit = max(((float(np.dot(p, prof)), n)
                                            for n, p in active.items()), default=(-1.0, None))
                        if hit is not None and hit_sim >= dd:
                            target = hit          # アクティブな既存人物の声だった → 合流
                            is_new = False
                        else:
                            self.n_anon += 1
                            target = f"人物{self.n_anon}"
                            self.profiles[target] = prof   # 新規人物（以後凍結）
                            self._active_keys.add(target)  # セッション中の新規人物は自動アクティブ
                            is_new = True
                        # 遡及置換は未確定キー(#ラベル)の昇格のみ。人物キーは絶対に書き換えない。
                        rename = ("#" + sp, target) if (prev is None or prev.startswith("#")) else None
                        self.sp_map[sp] = target
                        kind = "自動登録" if is_new else "合流"
                        self._note(kind, label=sp, name=target, rename=rename)
                        return target
                    self.pool.append(emb)
                    del self.pool[:-12]
        # 声紋で決められない（重なり/短い相槌/蓄積中）→ ラベルの直近判定に追従
        key = prev if prev is not None else "#" + sp
        self.sp_map[sp] = key
        self._note(kind, label=sp, **info)
        return key

    def enroll(self, label: str, name: str) -> str | None:
        """「1=松井」「人物2=田中」: 話者に名前を付ける（声の登録 or 既存人物のリネーム）.

        実名を付けたプロファイルのみ voices.json に永続化される（匿名「人物N」は
        そのセッション限り）。戻り値: 旧表示キー（過去のrecords付け替え用）。
        十分な音声がまだ無ければ None。
        """
        with self._lock:
            return self._enroll(str(label), name)

    def _enroll(self, label: str, name: str) -> str | None:
        if label in self.profiles:
            # 「人物1=松井」: 既存プロファイルのリネーム
            self.profiles[name] = self.profiles.pop(label)
            old = label
        else:
            cur = self.sp_map.get(label)
            if cur is not None and cur in self.profiles:
                # ラベルが（自動登録済みの）人物に対応済み → その人物に命名
                self.profiles[name] = self.profiles.pop(cur)
                old = cur
            else:
                # ラベルの直近声紋から新規登録
                embs = self.label_embs.get(label)
                if not embs:
                    return None
                prof = np.mean(embs, axis=0)
                self.profiles[name] = prof / np.linalg.norm(prof)
                old = cur if cur is not None else "#" + label
        # _active_keysの更新（旧キーが有効だったら新キーに引き継ぐ）
        if old in self._active_keys:
            self._active_keys.discard(old)
            self._active_keys.add(name)
        else:
            self._active_keys.add(name)   # 新規命名は自動的にアクティブ
        for k, v in list(self.sp_map.items()):
            if v == old:
                self.sp_map[k] = name
        if old in self.own_sims:
            self.own_sims[name] = self.own_sims.pop(old)
        if old != label:   # 「人物N=名前」のリネーム以外は、ラベル自体も対応づける
            self.sp_map[label] = name
        self._persist()
        return old

    def remap(self, src: str, dst: str) -> bool:
        """「fix 人物2=人物1」: srcをdstに統合（srcのプロファイルも削除し、復活を防ぐ）."""
        with self._lock:
            if src == dst:
                return False
            self.profiles.pop(src, None)   # 残すと同じ声が再びsrcと判定されて復活してしまう
            self.own_sims.pop(src, None)
            for k, v in list(self.sp_map.items()):
                if v == src:
                    self.sp_map[k] = dst
            self._active_keys.discard(src)
            self._persist()
            return True

    def activate(self, name: str) -> str | None:
        """プロファイルをこのセッションで有効化する.

        有効化されたプロファイルは _classify() の照合対象になる。
        既にセッション中に自動登録された人物Nが同一人物だった場合は自動マージし、
        マージされた旧キーを返す（rekey用）。マージなしならNone。
        """
        with self._lock:
            if name not in self.profiles:
                return None
            self._active_keys.add(name)
            prof = self.profiles[name]
            # セッション中の匿名人物Nに同一人物がいたらマージ
            for key in list(self._active_keys):
                if self.ANON.match(key) and key in self.profiles:
                    sim = float(np.dot(prof, self.profiles[key]))
                    if sim >= self.dedupe:
                        self.profiles.pop(key)
                        self._active_keys.discard(key)
                        self.own_sims.pop(key, None)
                        for k, v in list(self.sp_map.items()):
                            if v == key:
                                self.sp_map[k] = name
                        return key   # マージされた旧キー
            return None

    def deactivate(self, name: str) -> None:
        """プロファイルをこのセッションで無効化する（匿名人物Nは無効化不可）."""
        with self._lock:
            if not self.ANON.match(name):
                self._active_keys.discard(name)

    def active_profile_names(self) -> list[str]:
        """現在アクティブな名前付きプロファイルの一覧（UI表示用）."""
        return sorted(k for k in self._active_keys if not self.ANON.match(k) and k in self.profiles)

    def all_profile_names(self) -> list[str]:
        """voices.jsonに保存された全名前付きプロファイル（UI表示用）."""
        return sorted(k for k in self.profiles if not self.ANON.match(k))

    def _persist(self):
        named = {k: v.tolist() for k, v in self.profiles.items() if not self.ANON.match(k)}
        named["_model"] = self.model   # 声紋はモデル間で互換性がないため記録
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(named, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def stats(self) -> str:
        parts = []
        if self.embed_ms:
            a = np.array(self.embed_ms)
            parts.append(f"声紋計算 {len(a)}回 平均{a.mean():.0f}ms 最大{a.max():.0f}ms")
        if len(self.same_sims) >= 8 and len(self.diff_sims) >= 12:
            parts.append(f"部屋の声紋分布(参考): ラベル内{np.median(self.same_sims):.2f}"
                         f"/ラベル間{np.median(self.diff_sims):.2f}")
        if self.counts:
            order = ["声紋一致", "補正", "自動登録", "合流", "蓄積中", "未確定", "相槌追従",
                     "重なりスキップ", "声紋計算不可"]
            parts.append("判定内訳: " + " / ".join(
                f"{k}{self.counts[k]}" for k in order if self.counts.get(k)))
        return "、".join(parts) or "判定なし"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default="stt-rt-v4")
    ap.add_argument("--wav", default=None, help="指定で実マイクの代わりにファイル擬似ライブ")
    ap.add_argument("--play", action="store_true",
                    help="--wav使用時、注入と同時にスピーカーからも再生する（観戦用）")
    ap.add_argument("--join", action="store_true",
                    help="--wav使用時、再生しつつ自分のマイクも混ぜて参加する（イヤホン推奨。"
                         "wav終了後もマイクは生き続けるのでCtrl+Cで終了）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="保存先mdファイル（省略時 transcripts/日時.md）")
    ap.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")
    ap.add_argument("--no-vp", action="store_true", help="声紋照合を無効化（Sonioxのラベルをそのまま使う）")
    ap.add_argument("--voices", default="voices.json", help="声紋プロファイルの保存先(既定 voices.json)")
    ap.add_argument("--vp-model", default="redimnet", choices=["redimnet", "ecapa", "resemblyzer"],
                    help="声紋モデル(既定redimnet=2024年世代、実測の分離・通し精度とも最良。"
                         "読み込み失敗時は ecapa → resemblyzer へ自動フォールバック)")
    ap.add_argument("--vp-match", type=float, default=None,
                    help="即時判定のしきい値。省略時はモデル別の既定値"
                         "(redimnet 0.42 / ecapa 0.35 / resemblyzer 0.75)")
    ap.add_argument("--vp-no-auto", action="store_true",
                    help="未知の声の自動登録（匿名「人物N」）を無効化")
    ap.add_argument("--vp-debug", action="store_true", help="発話ごとの声紋判定の内訳を表示")
    ap.add_argument("--no-polish", action="store_true",
                    help="終了時の清書（非同期APIでの全体再処理）を行わない")
    ap.add_argument("--stt", default="soniox", choices=["soniox", "speechmatics"],
                    help="リアルタイムSTTの供給源。speechmaticsは要 SPEECHMATICS_API_KEY"
                         "（話者分離の評判が良い代替。声紋層など他の機能は不変）")
    ap.add_argument("--port", type=int, default=8231,
                    help="UIサーバーのポート番号（ブラウザからの話者リネームに必要。0で無効）")
    ap.add_argument("--agent", action="store_true",
                    help="AIエージェント（ファシリテーター）を有効化。OPENAI_API_KEYが必要。"
                         "Realtime API v2 WebSocketで会議に参加する")
    ap.add_argument("--agent-voice", default="alloy",
                    help="AIエージェントの声（alloy/ash/ballad/coral/echo/sage/shimmer/verse）")
    ap.add_argument("--agent-trigger", type=int, default=_AGENT_TRIGGER,
                    help=f"AIの応答を検討する発話間隔（既定{_AGENT_TRIGGER}）")
    args = ap.parse_args(argv)
    _serve = args.port > 0

    load_env()   # .env からAPIキーを読み込み（export済みの値が優先）
    if args.wav and not os.path.exists(args.wav):
        raise SystemExit(f"音声ファイルがありません: {args.wav}\n"
                         "（テスト音声は scripts/make_overlap_testset.py 等で先に生成してください）")

    api_key = os.environ.get("SONIOX_API_KEY")
    sm_key = os.environ.get("SPEECHMATICS_API_KEY")
    if args.stt == "speechmatics":
        if not sm_key:
            raise SystemExit("環境変数 SPEECHMATICS_API_KEY を設定してください"
                             "（https://portal.speechmatics.com/settings/api-keys）")
    elif not api_key:
        raise SystemExit("環境変数 SONIOX_API_KEY を設定してください（https://console.soniox.com）")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("uv add websockets を実行してください")

    if args.stt == "speechmatics":
        ws_url = SM_WS_URL
        ws_headers = {"Authorization": f"Bearer {sm_key}"}
        start_msg = {
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": SR},
            "transcription_config": {
                "language": args.lang,
                "operating_point": "enhanced",
                "diarization": "speaker",
                "enable_partials": True,
                "max_delay": 1.2,
                "conversation_config": {"end_of_utterance_silence_trigger": 0.8},
            },
        }
    else:
        ws_url = WS_URL
        ws_headers = None
        start_msg = {
            "api_key": api_key,
            "model": args.model,
            "language_hints": [args.lang],
            "enable_speaker_diarization": True,
            "enable_endpoint_detection": True,
            "audio_format": "pcm_s16le",
            "sample_rate": SR,
            "num_channels": 1,
        }

    started = datetime.datetime.now()
    if args.out:
        out_path = args.out
    else:
        os.makedirs("transcripts", exist_ok=True)
        out_path = os.path.join("transcripts", started.strftime("%Y-%m-%d_%H%M") + ".md")
    html_path = os.path.splitext(out_path)[0] + ".html"
    diag_path = os.path.splitext(out_path)[0] + ".diag.jsonl"   # 発話ごとの判定根拠(劣化解析用)
    turns_path = os.path.splitext(out_path)[0] + ".turns.jsonl"  # das(議論支援)連携用

    # --- 状態 ---
    names: dict[str, str] = {}          # 表示キー -> 別名（声紋OFF時の命名用）
    colors: dict[str, str] = {}         # 表示キー -> ANSI色（出現順に割当）
    records: list[dict] = []            # 確定発話 {"ms", "speaker", "text"}
    state_lock = threading.Lock()

    tracker: VoiceProfiles | None = None
    if not args.no_vp:
        print("# 声紋モデルを読み込み中…", flush=True)
        for model in dict.fromkeys([args.vp_model, "ecapa", "resemblyzer"]):
            try:
                tracker = VoiceProfiles(path=args.voices, thresh=args.vp_match,
                                        auto=not args.vp_no_auto, model=model)
                if model != args.vp_model:
                    print(f"# 注意: {args.vp_model} を読み込めなかったため {model} で動作します"
                          f"（依存: uv add speechbrain torchaudio / redimnetは初回ネット接続必要）",
                          flush=True)
                print(f"# 声紋モデル: {model}", flush=True)
                break
            except Exception as e:   # 依存欠如(ImportError)もDL失敗等も次の候補へ
                print(f"#   {model}: 読み込み失敗 ({type(e).__name__})", flush=True)
                continue
        if tracker is None:
            print("# 警告: 声紋照合がOFFです！ 依存が未導入のため人物の確定・補正は行われません。", flush=True)
            print("#   有効化するには: uv add speechbrain torchaudio  →  再起動", flush=True)
        elif tracker.profiles:
            print(f"# 声紋プロファイル: {', '.join(tracker.profiles)}（{args.voices}）", flush=True)
        else:
            print(f"# 声紋プロファイル: なし。未知の声は「人物N」として自動追跡、"
                  f"「1=松井」で実名化すると次回から自動表示（{args.voices}）", flush=True)

    # --- AIエージェント ---
    agent: RealtimeAgent | None = None
    _agent_oai_key = os.environ.get("OPENAI_API_KEY", "")
    if args.agent:
        if not _agent_oai_key:
            print("# AI Agent: OPENAI_API_KEY が未設定です。--agent は無効になります。", flush=True)
        else:
            agent = RealtimeAgent(api_key=_agent_oai_key, voice=args.agent_voice,
                                  mode="facilitator", trigger_n=args.agent_trigger)

    pcm_buf = bytearray()               # 送信済み音声の全バッファ（声紋切り出し用, 16bit）
    buf_lock = threading.Lock()

    def disp_name(key) -> str:
        key = str(key)
        if key in names:
            return names[key]
        return f"話者{key[1:]}" if key.startswith("#") else key

    def key_for_label(sp) -> str:
        sp = str(sp)
        if tracker is not None and sp in tracker.sp_map:
            return tracker.sp_map[sp]
        return "#" + sp

    def color_of(key) -> str:
        key = str(key)
        if key not in colors:
            colors[key] = PALETTE[len(colors) % len(PALETTE)]
        return colors[key]

    def rekey(old: str, new: str):
        """表示キーの付け替え: recordsと色を一括移行（話者一覧に旧キーの幽霊を残さない）."""
        with state_lock:
            for r in records:
                if r.get("speaker") == old:
                    r["speaker"] = new
            if old in colors:
                colors.setdefault(new, colors.pop(old))

    def add_sys(ms, text: str):
        """システムイベント（補正・自動登録・命名・統合）を議事録のタイムラインに残す."""
        with state_lock:
            records.append({"ms": ms, "sys": text})

    global _SYS_HOOK

    def _sys_hook(text: str) -> None:   # das介入をライブHTML/MDに反映
        add_sys(None, text)
        save()
    _SYS_HOOK = _sys_hook

    def write_md(recs=None, path=None):
        with state_lock:
            rs = records if recs is None else recs
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            lines = [
                f"# 議事録 {started.strftime('%Y-%m-%d %H:%M')}",
                "",
                "話者: " + (", ".join(disp_name(s) for s in speakers) or "（未検出）"),
                "",
            ]
            for r in rs:
                if "sys" in r:
                    lines.append(f"> [{fmt_ts(r['ms'])}] {r['sys']}")
                    continue
                mark = " ⚡" if r.get("vp") == "補正" else ""
                lines.append(f"- **[{fmt_ts(r['ms'])}] {disp_name(r['speaker'])}{mark}**: {r['text']}")
            dst = path or out_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, dst)

    def write_html(live: bool = True, recs=None, path=None, status=None):
        import html as _html
        with state_lock:
            rs = records if recs is None else recs
            parts = []
            for r in rs:
                if "sys" in r:
                    parts.append(f'<div class="sys">⚙ {_html.escape(r["sys"])}</div>')
                    continue
                sp = str(r["speaker"])
                color_of(sp)
                idx = list(colors).index(sp)
                c = HTML_PALETTE[idx % len(HTML_PALETTE)]
                badge = ""
                if r.get("vp") == "補正":
                    note = _html.escape(r.get("note", ""))
                    badge = f'<span class="badge" title="{note}">⚡声紋補正</span>'
                parts.append(
                    f'<div class="u"><span class="ts">{fmt_ts(r["ms"])}</span>'
                    f'<span class="who" style="color:{c}">{_html.escape(disp_name(sp))}</span>'
                    f'{_html.escape(r["text"])}{badge}</div>'
                )
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            # サイドバー話者パネル: 確定済み話者（人物N＋実名）はリネーム可能、未確定(話者N)は表示のみ
            sp_tags = []
            for s in speakers:
                dn = _html.escape(disp_name(s))
                idx_s = list(colors).index(s) if s in colors else 0
                c = HTML_PALETTE[idx_s % len(HTML_PALETTE)]
                # #N(話者N)は未確定なのでリネーム不可。人物N・実名は確定済みなのでリネーム可能
                is_renameable = _serve and tracker is not None and not s.startswith("#")
                if is_renameable:
                    lbl = s  # enroll()は"人物1"でも"松井"でも受け付ける
                    for _l, _k in tracker.sp_map.items():
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
            # プロファイル一覧パネル（voices.jsonに保存済みのプロファイルをトグル表示）
            profile_panel = ''
            if _serve and tracker is not None:
                all_names = tracker.all_profile_names()
                if all_names:
                    active_names = set(tracker.active_profile_names())
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
            # 発言量統計パネル（発話時間・文字数・発話回数の割合）
            stats_panel = ''
            talk_rs = [r for r in rs if "speaker" in r and r.get("text")]
            if talk_rs:
                # 話者ごとに集計
                sp_dur: dict[str, float] = {}   # 発話時間(秒)
                sp_chars: dict[str, int] = {}   # 文字数
                sp_turns: dict[str, int] = {}   # 発話回数
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
                # 発話時間順でソート（最も話した人が上）
                ranked = sorted(sp_dur.keys(), key=lambda s: sp_dur[s], reverse=True)

                def _bar_rows(data, total, unit=""):
                    rows = []
                    for s in ranked:
                        v = data.get(s, 0)
                        pct = v / total * 100 if total else 0
                        idx_s = list(colors).index(s) if s in colors else 0
                        c = HTML_PALETTE[idx_s % len(HTML_PALETTE)]
                        dn = _html.escape(disp_name(s))
                        # 短い名前の先頭2文字
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
                if total_dur > 0.5:   # 発話時間は十分なデータがある時だけ
                    groups.append(f'<div class="stats-group">'
                                  f'<div class="stats-label">発話時間</div>'
                                  + _bar_rows(sp_dur, total_dur) + '</div>')
                groups.append(f'<div class="stats-group">'
                              f'<div class="stats-label">文字数</div>'
                              + _bar_rows(sp_chars, total_chars) + '</div>')
                groups.append(f'<div class="stats-group">'
                              f'<div class="stats-label">発話回数</div>'
                              + _bar_rows(sp_turns, total_turns) + '</div>')
                stats_panel = ('<div class="stats-section">'
                               '<p class="sidebar-title">発言量</p>'
                               + ''.join(groups) + '</div>')
            # 論点パネル
            topics_panel = ''
            with topics_lock:
                if topics:
                    items = []
                    for t in topics:
                        tt = _html.escape(t.get("topic", ""))
                        ts = _html.escape(t.get("speaker", ""))
                        items.append(f'<div class="topic-item">'
                                     f'<div class="topic-text">{tt}</div>'
                                     f'<div class="topic-by">{ts}</div></div>')
                    topics_panel = ('<div class="topics-section">'
                                   '<p class="sidebar-title">論点</p>'
                                   + ''.join(items) + '</div>')
            # AIエージェントパネル
            agent_panel = ''
            if agent is not None:
                cur_mode = agent.mode
                if agent._connected:
                    conn = '接続中'
                elif agent._conn_error:
                    conn = f'エラー: {_html.escape(agent._conn_error)}'
                else:
                    conn = '未接続'
                # モード選択ボタン
                mode_btns = []
                for m, lbl in [("off", "OFF"), ("facilitator", "進行役"),
                               ("conversation", "会話")]:
                    cls = "agent-mode-btn active" if m == cur_mode else "agent-mode-btn"
                    mode_btns.append(f'<button class="{cls}" data-mode="{m}" '
                                     f'onclick="setAgentMode(this)">{lbl}</button>')
                # 声選択
                voice_opts = []
                for v in AGENT_VOICES:
                    sel = 'selected' if v == agent.voice else ''
                    voice_opts.append(f'<option value="{v}" {sel}>{v}</option>')
                # トリガー間隔(facilitatorのみ)
                trigger_val = agent.trigger_n
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
                title=started.strftime("%Y-%m-%d %H:%M"),
                status=status or ('<span class="live">● ライブ（2秒ごと自動更新）</span>'
                                  if live else "終了"),
                speaker_panel=speaker_panel,
                profile_panel=profile_panel,
                stats_panel=stats_panel,
                topics_panel=topics_panel,
                agent_panel=agent_panel,
                body="\n".join(parts) or '<p class="meta">（まだ発話なし）</p>',
            )
            dst = path or html_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(doc)
            os.replace(tmp, dst)

    def write_turns(recs=None, path=None):
        """discussion-support(das)のUtteranceスキーマでJSONL出力.

        `das run-session 日時.turns.jsonl` がそのまま読める形式。saveごとに全体を
        書き直すので、後からの実名化(1=松井)や統合(fix)が過去の行にも反映される。
        """
        with state_lock:
            rs = records if recs is None else recs
            lines = []
            tid = 0
            for r in rs:
                if "speaker" not in r or not r.get("text"):
                    continue
                tid += 1
                lines.append(json.dumps({"turn_id": tid, "speaker": disp_name(r["speaker"]),
                                         "text": r["text"], "ms": r.get("ms"),
                                         "end_ms": r.get("end_ms")},
                                        ensure_ascii=False))
            dst = path or turns_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            os.replace(tmp, dst)

    def save(live: bool = True):
        write_md()
        write_html(live)
        write_turns()

    # --- 論点抽出（非同期バックグラウンド処理）---
    topics: list[dict] = []          # {"topic": "...", "speaker": "...", "ms": ...}
    topics_lock = threading.Lock()
    _topic_cursor = 0                # recordsの何番目まで処理済みか
    _TOPIC_WINDOW = 10               # LLMに渡す直近発話数
    _TOPIC_TRIGGER = 5               # 新発話がこの数たまったらLLM呼び出し
    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    _oai_model = os.environ.get("OPENAI_MODEL_FAST", "gpt-5-mini")

    def _topic_worker():
        nonlocal _topic_cursor
        while not stop.is_set():
            time.sleep(3)
            if not _oai_key:
                continue
            with state_lock:
                talk_rs = [r for r in records if "speaker" in r and r.get("text")]
            n = len(talk_rs)
            if n - _topic_cursor < _TOPIC_TRIGGER:
                continue
            # 直近ウィンドウを取得
            window = talk_rs[max(0, n - _TOPIC_WINDOW):]
            utts = [{"speaker": disp_name(r["speaker"]), "text": r["text"]} for r in window]
            with topics_lock:
                existing = [t["topic"] for t in topics]
            new_topics = _extract_topics(utts, existing, _oai_key, _oai_model)
            if new_topics:
                ms = window[-1].get("ms")
                with topics_lock:
                    for t in new_topics:
                        if isinstance(t, dict) and "topic" in t:
                            topics.append({"topic": t["topic"],
                                           "speaker": t.get("speaker", "?"),
                                           "ms": ms})
                save()
                for t in new_topics:
                    if isinstance(t, dict) and "topic" in t:
                        print_line(f"# 💡論点: {t['topic']}（{t.get('speaker', '?')}）")
            _topic_cursor = n

    # --- AIエージェント: コールバック + ワーカースレッド ---
    def _on_agent_text(text: str):
        """E: AIの生成テキストをrecordsに直接挿入（STTバイパス）."""
        with state_lock:
            records.append({"ms": None, "end_ms": None,
                            "speaker": AGENT_SPEAKER, "text": text.strip()})
            color_of(AGENT_SPEAKER)
        if ON_UTTERANCE is not None:
            try:
                ON_UTTERANCE("AI", text.strip())
            except Exception:
                pass
        print_line(f"\x1b[96m[AI] AI\x1b[0m: {text.strip()}")
        save()

    _agent_cursor = 0
    _last_utt_time = [time.monotonic()]   # mutableで非ローカル参照

    def _agent_worker():
        """バックグラウンドでAI応答のトリガーを管理."""
        nonlocal _agent_cursor
        while not stop.is_set():
            time.sleep(0.5)
            if agent is None or not agent._connected or not agent.enabled:
                continue
            with state_lock:
                talk_rs = [r for r in records
                           if "speaker" in r and r.get("text")
                           and r.get("speaker") != AGENT_SPEAKER]
            n = len(talk_rs)
            # クールダウン中: カーソルを止めてバックログを溜める。
            # flush()がエコーを除去済みなので、recordsには人間の発話だけ残る。
            # クールダウン終了後にまとめて処理し、ここでも類似度チェックする（2重フィルタ）。
            if agent.in_echo_cooldown:
                continue
            if n > _agent_cursor:
                _last_utt_time[0] = time.monotonic()
                fed = 0
                for r in talk_rs[_agent_cursor:]:
                    _sp = r.get("speaker", "")
                    _txt = r.get("text", "")
                    # 2重フィルタ: flush()を通過したレコードの最終エコーチェック
                    _sim = agent._best_similarity(_txt)
                    if (agent.is_ai_echo(_sp) and _sim > 0.25) or _sim > 0.40:
                        if args.vp_debug:
                            print_line(f"# worker: エコー除去 sim={_sim:.2f}"
                                       f" sp={_sp} ({_txt[:30]}...)")
                        continue
                    agent.feed(disp_name(_sp), _txt)
                    fed += 1
                _agent_cursor = n
                if fed == 0:
                    continue  # 全てエコーだった場合はトリガーしない
            # モード別トリガー判定
            if agent.mode == "conversation":
                # 会話モード: 新発話があったら即trigger（0.5秒以内）
                if agent.pending_count > 0:
                    agent.trigger()
            else:
                # ファシリテーター: N発話蓄積 or 沈黙
                if agent.pending_count >= agent.trigger_n:
                    agent.trigger()
                elif (agent.pending_count > 0
                      and time.monotonic() - _last_utt_time[0] > _AGENT_SILENCE):
                    agent.trigger()

    # --- UIサーバー（ブラウザからの話者リネーム用）---
    _httpd = None
    if _serve:
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    try:
                        with open(html_path, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(content)
                    except FileNotFoundError:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        self.wfile.write("<p>準備中…</p>".encode())
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == "/rename":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    label = str(body.get("label", ""))
                    name = str(body.get("name", ""))
                    if not label or not name:
                        self._json(400, {"error": "label と name を指定してください"})
                        return
                    if tracker is not None:
                        old = tracker.enroll(label, name)
                        if old is None:
                            self._json(400, {"error": f"話者{label}の音声がまだ足りません"})
                            return
                        rekey(old, name)
                        add_sys(None, f"「{name}」の声を登録（次回の会議から自動表示）")
                        save()
                        print_line(f"# {name} の声を登録しました（UIから）")
                    else:
                        with state_lock:
                            names["#" + label] = name
                        save()
                        print_line(f"# 話者{label} → {name}（UIから）")
                    self._json(200, {"ok": True, "name": name})
                elif self.path == "/activate":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    name = str(body.get("name", ""))
                    active = bool(body.get("active", True))
                    if not name:
                        self._json(400, {"error": "name を指定してください"})
                        return
                    if tracker is None:
                        self._json(400, {"error": "声紋照合が無効です"})
                        return
                    if active:
                        merged = tracker.activate(name)
                        if merged is not None:
                            rekey(merged, name)
                            add_sys(None, f"「{name}」を有効化（{merged}と統合）")
                            print_line(f"# {name} を有効化（{merged}と統合、UIから）")
                        else:
                            print_line(f"# {name} を有効化（UIから）")
                        save()
                    else:
                        tracker.deactivate(name)
                        print_line(f"# {name} を無効化（UIから）")
                        save()
                    self._json(200, {"ok": True, "name": name, "active": active})
                elif self.path == "/agent":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    if agent is None:
                        self._json(400, {"error": "AIエージェントが無効です（--agent で起動してください）"})
                        return
                    mode = body.get("mode")
                    voice = body.get("voice")
                    trigger_n = body.get("trigger_n")
                    if trigger_n is not None:
                        trigger_n = int(trigger_n)
                    agent.apply_config(mode=mode, voice=voice, trigger_n=trigger_n)
                    print_line(f"# AI Agent 設定変更: mode={agent.mode} voice={agent.voice}"
                               f" trigger={agent.trigger_n}（UIから）")
                    save()   # HTMLを即時更新（meta-refreshで古い状態が表示されるのを防止）
                    self._json(200, {"ok": True, "mode": agent.mode,
                                     "voice": agent.voice, "trigger_n": agent.trigger_n})
                else:
                    self.send_error(404)

            def _json(self, code, data):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

            def log_message(self, format, *args):
                pass

        try:
            _httpd = HTTPServer(("127.0.0.1", args.port), _Handler)
            threading.Thread(target=_httpd.serve_forever, daemon=True).start()
        except OSError as e:
            print(f"# 警告: UIサーバーをポート{args.port}で起動できません ({e})", flush=True)
            _serve = False

    def print_line(text: str):
        sys.stdout.write(CLEAR_LINE + text + "\n")
        sys.stdout.flush()

    def show_partial(sp, text: str):
        if not text.strip():
            sys.stdout.write(CLEAR_LINE)
        else:
            cols = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            line = f"{disp_name(key_for_label(sp))}: {text.strip()}"
            sys.stdout.write(CLEAR_LINE + DIM + line[-(cols - 2):] + RESET)
        sys.stdout.flush()

    # --- 音声入力（マイク or ファイル）→ audio_q ---
    audio_q: "queue.Queue[bytes | None]" = queue.Queue()
    stop = threading.Event()

    def from_mic():
        import sounddevice as sd

        def cb(indata, frames, t, status):
            pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()
            audio_q.put(pcm)
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            device=args.device, callback=cb, blocksize=int(SR * 0.1)):
            while not stop.is_set():
                time.sleep(0.1)
        audio_q.put(None)

    def from_wav():
        import librosa
        y, _ = librosa.load(args.wav, sr=SR)
        step = int(SR * 0.12)
        out = mic = None
        if args.play or args.join:
            import sounddevice as sd
            out = sd.OutputStream(samplerate=SR, channels=1, dtype="float32")
            out.start()
        if args.join:
            import sounddevice as sd
            mic = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=step)
            mic.start()
        i = 0
        while not stop.is_set():
            chunk = np.clip(y[i:i + step], -1, 1).astype("float32") if i < len(y) else                 np.zeros(0, dtype="float32")
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            i += step
            if i - step >= len(y) and mic is None:
                break   # wav終了(参加モードでなければここで終わり)
            if mic is not None:
                mdata, _ = mic.read(step)        # マイク読みが実時間ペースを刻む
                mix = np.clip(chunk + mdata[:, 0], -1, 1)
            else:
                mix = chunk
            audio_q.put((mix * 32767).astype("<i2").tobytes())
            if out is not None:
                out.write(chunk.reshape(-1, 1))   # 自分の声は再生しない(ハウリング防止)
                if mic is None:
                    continue                       # 再生がペースを刻む
            if mic is None and out is None:
                time.sleep(0.12)
        for s in (out, mic):
            if s is not None:
                s.stop()
                s.close()
        audio_q.put(None)

    # --- 実行中コマンド ---
    def key_of(tok: str) -> str:
        """コマンド引数を表示キーへ: 人物名はそのまま、数字はそのラベルの現在の表示先."""
        if tracker is not None:
            if tok in tracker.profiles:
                return tok
            if tok in tracker.sp_map:
                return tracker.sp_map[tok]
        return "#" + tok

    def stdin_commands():
        while not stop.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            mfix = re.match(r"^\s*fix\s+(\S+)\s*=\s*(\S+)\s*$", line)
            m = re.match(r"^\s*(\S+?)\s*=\s*(.+?)\s*$", line)
            if mfix:
                src, dst = key_of(mfix.group(1)), key_of(mfix.group(2))
                if tracker is not None:
                    tracker.remap(src, dst)
                rekey(src, dst)
                add_sys(None, f"{disp_name(src)} を {disp_name(dst)} に統合（手動fix）")
                save()
                print_line(f"# {disp_name(src)} を {disp_name(dst)} に統合しました（過去の発言も修正済み）")
            elif m:
                label, name = m.group(1), m.group(2)
                if tracker is not None:
                    old = tracker.enroll(label, name)
                    if old is None:
                        print_line(f"# 話者{label}の音声がまだ足りません（1秒以上話してから再実行）")
                        continue
                    rekey(old, name)
                    add_sys(None, f"「{name}」の声を登録（次回の会議から自動表示）")
                    save()
                    print_line(f"# {name} の声を登録しました（過去の発言も置換、次回の会議から自動表示）")
                else:
                    with state_lock:
                        names["#" + label] = name
                    save()
                    print_line(f"# 話者{label} → {name}（過去の発言も置換済み）")
            elif line.strip():
                print_line("# コマンド: 「1=松井」(声を登録) / 「fix 2=1」「fix 人物2=人物1」(統合) / Ctrl+Cで終了")

    print(f"# {args.stt} に接続中…", flush=True)
    with connect(ws_url, additional_headers=ws_headers) as ws:
        ws.send(json.dumps(start_msg))
        threading.Thread(target=from_wav if args.wav else from_mic, daemon=True).start()
        threading.Thread(target=stdin_commands, daemon=True).start()
        if _oai_key:
            threading.Thread(target=_topic_worker, daemon=True).start()
            print("# 論点抽出: 有効（5発話ごとにLLMで分析）", flush=True)
        else:
            print("# 論点抽出: 無効（OPENAI_API_KEYが未設定）", flush=True)
        if agent is not None:
            agent.on_ai_utterance = _on_agent_text
            agent.connect()
            threading.Thread(target=_agent_worker, daemon=True).start()
            print(f"# AI Agent: mode={agent.mode} voice={agent.voice}"
                  f" trigger={agent.trigger_n}（ブラウザから変更可能）", flush=True)

        def sender():
            seq = 0
            while True:
                pcm = audio_q.get()
                if pcm is None:   # 終端
                    if args.stt == "speechmatics":
                        ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": seq}))
                    else:
                        ws.send("")
                    break
                with buf_lock:
                    pcm_buf.extend(pcm)   # STTの時刻軸と完全一致する位置で蓄積
                # AEC: AI再生音声をマイク入力から減算してからSTTへ送信
                if agent is not None and agent._connected:
                    try:
                        mic_f = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                        cleaned = agent.cancel_echo(mic_f)
                        pcm = (np.clip(cleaned, -1, 1) * 32767).astype("<i2").tobytes()
                    except Exception:
                        pass  # AEC失敗時は元のpcmをそのまま送信
                ws.send(pcm)
                seq += 1
        threading.Thread(target=sender, daemon=True).start()

        save()
        print("# 開始。話してください（「1=松井」で声を登録 / Ctrl+Cで終了）", flush=True)
        print(f"# 保存先: {out_path}", flush=True)
        print(f"# ブラウザ表示: open {html_path}（ライブ中は2秒ごと自動更新）\n", flush=True)
        if not args.no_open:
            import webbrowser
            if _serve:
                webbrowser.open(f"http://127.0.0.1:{args.port}/")
            else:
                webbrowser.open("file://" + os.path.abspath(html_path))

        cur_speaker = None
        cur_text = ""
        cur_ms: int | None = None
        cur_end: int | None = None
        recent_segs: list[tuple] = []   # (start, end, ラベル) 直近の確定発話（重なり検出用）

        def overlaps_other(start, end, label) -> bool:
            if start is None or end is None:
                return False
            return any(l != label and min(e, end) - max(s, start) > 0
                       for s, e, l in recent_segs)

        def flush():
            nonlocal cur_text, cur_ms, cur_end
            if cur_text.strip():
                label = str(cur_speaker)
                if tracker is not None:
                    if cur_ms is not None and cur_end is not None and cur_end > cur_ms:
                        with buf_lock:
                            seg = bytes(pcm_buf[cur_ms * 32: cur_end * 32])  # 16サンプル/ms×2byte
                        wav = np.frombuffer(seg, dtype="<i2").astype(np.float32) / 32768.0
                    else:
                        wav = np.zeros(0, dtype=np.float32)
                    sp_id = tracker.classify(wav, cur_speaker,
                                             overlapped=overlaps_other(cur_ms, cur_end, label))
                    d = tracker.last
                    rec_extra = {}
                    if d and d["kind"] == "補正":
                        note = (f"声紋でラベル{d['label']}の取り違えを修正"
                                f"（類似{d['sim']:.2f}、放置なら{disp_name(d['prev'])}の発言になっていた）")
                        rec_extra = {"vp": "補正", "note": note}
                        print_line(f"# ⚡補正: {note}")
                    elif d and d["kind"] == "自動登録":
                        if d["rename"]:   # 「#ラベル→人物」の昇格のみ遡及置換（人物キーは不変）
                            rekey(*d["rename"])
                        add_sys(cur_ms, f"この声を「{d['name']}」として追跡開始"
                                        f"（実名にするには {d['label']}=名前）")
                        print_line(f"# この声を「{d['name']}」として追跡します"
                                   f"（実名にするには {d['label']}=名前 と入力）")
                    elif d and d["kind"] == "合流":
                        if d["rename"]:
                            rekey(*d["rename"])
                        # 既存人物への合流はターミナルにのみ軽く表示（議事録には載せない）
                        if args.vp_debug:
                            print_line(f"# 合流: ラベル{d['label']}→{d['name']}")
                    elif args.vp_debug and d:
                        extra = f" 類似{d['sim']:.2f}({d['name']})" if "sim" in d else ""
                        print_line(f"# vp判定[{d['kind']}]{extra}")
                else:
                    sp_id = "#" + str(cur_speaker)
                    rec_extra = {}
                # --- エコー安全網: 2層判定 ---
                # マイクは常時オン（人間の割り込みを取得するため）。
                # AI音声のエコーだけをテキスト+スピーカーIDで選別除去する。
                # 注意: 日本語テキスト同士は助詞・語尾の共通で無関係でも
                #        trigram Jaccard 0.15〜0.25 になるため閾値は高めに設定。
                if agent is not None:
                    sim = agent._best_similarity(cur_text)
                    _echo = False
                    _echo_reason = ""

                    # 層1: 既知AIエコースピーカー + 中程度の類似度（0.25）
                    if agent.is_ai_echo(sp_id) and sim > 0.25:
                        _echo = True
                        _echo_reason = f"既知AIスピーカー({sp_id})"
                    # 層2: 強いテキスト類似度（0.40）— 常時有効
                    # 真のエコー（STTがAI音声を文字起こし）なら0.5以上になる。
                    # 0.40でマーク＋除去。誤マーク防止のため以前の0.28から引き上げ。
                    elif sim > 0.40:
                        _echo = True
                        _echo_reason = "テキスト類似度"
                        if agent.in_echo_cooldown:
                            agent.mark_ai_echo(sp_id)

                    if _echo:
                        if args.vp_debug:
                            print_line(f"# エコー除去[{_echo_reason}] sim={sim:.2f}:"
                                       f" sp={sp_id} ({cur_text.strip()[:40]}...)")
                        cur_text = ""
                        cur_ms = None
                        cur_end = None
                        return
                if cur_ms is not None and cur_end is not None:
                    recent_segs.append((cur_ms, cur_end, label))
                    del recent_segs[:-12]
                if tracker is not None and tracker.last is not None:
                    # 診断ログ: 後から「いつ・なぜ判定が崩れたか」を解析するための1行JSON
                    try:
                        with open(diag_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"ms": cur_ms, "end": cur_end, "label": label,
                                                "key": sp_id, **tracker.last},
                                               ensure_ascii=False, default=str) + "\n")
                    except OSError:
                        pass
                with state_lock:   # colorsの変更も保存処理との競合を避けるためロック内で
                    records.append({"ms": cur_ms, "end_ms": cur_end,
                                    "speaker": sp_id, "text": cur_text.strip(),
                                    **rec_extra})
                    c = color_of(sp_id)
                if ON_UTTERANCE is not None:
                    try:
                        ON_UTTERANCE(disp_name(sp_id), cur_text.strip())
                    except Exception:
                        pass   # das側の例外で文字起こしを止めない
                print_line(f"{c}[{fmt_ts(cur_ms)}] {disp_name(sp_id)}{RESET}: {cur_text.strip()}")
                save()
            cur_text = ""
            cur_ms = None
            cur_end = None

        try:
            while True:
                res = json.loads(ws.recv())
                if args.stt == "speechmatics":
                    res = sm_to_res(res, args.lang)
                if res.get("error_code") is not None:
                    print_line(f"# エラー: {res['error_code']} - {res.get('error_message')}")
                    break
                partial = ""
                partial_sp = cur_speaker
                for token in res.get("tokens", []):
                    text = token.get("text") or ""
                    if text == "<end>":
                        flush()
                        continue
                    if not text:
                        continue
                    if token.get("is_final"):
                        sp = token.get("speaker")
                        if sp != cur_speaker:
                            flush()
                            cur_speaker = sp
                        if cur_ms is None:
                            cur_ms = token.get("start_ms")
                        if token.get("end_ms") is not None:
                            cur_end = token["end_ms"]
                        cur_text += text
                    else:
                        partial += text
                        partial_sp = token.get("speaker") or partial_sp
                show_partial(partial_sp if partial else cur_speaker, cur_text + partial)
                if res.get("finished"):
                    flush()
                    print_line("# 終了")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            globals()["_SYS_HOOK"] = None
            stop.set()
            if agent is not None:
                agent.close()
            flush()
            save(live=False)
            if tracker is not None:
                print_line(f"# レイテンシ統計: {tracker.stats()}")
            print_line(f"# 議事録を保存しました: {out_path} / {html_path}")
            if len(pcm_buf) > SR * 2 * 10:
                # 録音を保存（清書の再実験・診断用。*.wavはgitignore済み）
                wav_path = os.path.splitext(out_path)[0] + ".wav"
                try:
                    with open(wav_path, "wb") as f:
                        f.write(_wav_bytes(bytes(pcm_buf)))
                    print_line(f"# 録音を保存しました: {wav_path}")
                except OSError as e:
                    print_line(f"# 録音保存に失敗: {e}")
            # 清書: RT分離は高速応酬で崩れる(実測)ため、全文脈の非同期再処理で最終版を作る
            if not args.no_polish and not api_key and len(pcm_buf) > SR * 2 * 10:
                print_line("# 清書はスキップ（SONIOX_API_KEY未設定。清書はSoniox非同期APIを使用）")
            if not args.no_polish and api_key and len(pcm_buf) > SR * 2 * 10:
                try:
                    recs = polish(api_key, bytes(pcm_buf), args.lang, tracker, log=print_line)
                    fmd = os.path.splitext(out_path)[0] + ".final.md"
                    fht = os.path.splitext(out_path)[0] + ".final.html"
                    write_md(recs, fmd)
                    write_html(live=False, recs=recs, path=fht, status="清書（非同期再処理済み）")
                    write_turns(recs, os.path.splitext(out_path)[0] + ".final.turns.jsonl")
                    print_line(f"# 清書版を保存しました: {fmd} / {fht}")
                    if not args.no_open:
                        import webbrowser
                        webbrowser.open("file://" + os.path.abspath(fht))
                except KeyboardInterrupt:
                    print_line("# 清書をスキップしました")
                except Exception as e:
                    print_line(f"# 清書に失敗しました: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
