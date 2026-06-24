"""リアルタイム議事録モジュール定数."""
from __future__ import annotations

import re

SR = 16000

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

_DRIFT_PROMPT = """\
会議の論点と直近の発話を比較し、脱線しているか判定してください。

## 論点
{topics}

## 直近の発話
{utterances}

## 判定基準
- 発話が論点と無関係な話題（雑談、私的な話題など）→ drift=true
- 論点に関連する議論の展開・深掘り → drift=false
- 会議開始時の挨拶・自己紹介・進行の発言（「こんにちは」「よろしく」「始めましょう」等）
  → drift=false（脱線ではない）

JSON1つのみ出力。形式: {{"drift": true/false, "reason": "10字以内"}}"""

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

介入する場合は、前置きや「（介入）」のような記号を付けず、
本題の発言だけを話してください。
介入が不要だと判断した場合は、「（介入不要）」とだけ返してください。"""

_PROMPT_CONVERSATION = """\
あなたは会議に参加しているAIアシスタントです。
参加者と自然に会話してください。質問されたら必ず答えてください。
簡潔に、日本語で返答してください（15秒以内に収まる長さ）。
会議の文脈を踏まえた上で、役に立つ回答を心がけてください。"""

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"
AGENT_SPEAKER = "ファシリテーター"   # recordsに使うスピーカーキー
# =====================================================================
# タイミング定数（介入・割り込みロジック）
# ---------------------------------------------------------------------
# 関係の概要:
#   - ファシリテーターの通常トリガーは「N発話到達」または「一定時間の沈黙」。
#     debateモードはPartner会話が主なので沈黙閾値を長めにする。
#   - 並列ドリフトチェッカーは、ウォームアップ後、INTERVAL発話ごとに
#     直近WINDOW発話を見て脱線判定する。
#   - 介入不要後にデッドエアになったら STALL_SILENCE 秒で一押し（COOLDOWNで抑制）。
#   - エコーウィンドウ = AI発話終了後 ECHO_COOLDOWN 秒。この間はトリガー抑止＆
#     テキスト類似エコー除去を適用する。
# =====================================================================

# --- ファシリテーターの通常トリガー ---
_AGENT_TRIGGER = 10           # N発話ごとに応答検討(facilitator)
_AGENT_SILENCE = 5.0          # N秒沈黙で応答検討(facilitator, Partnerなし)
_AGENT_DEBATE_SILENCE = 15.0  # N秒沈黙で応答検討(debate — Partner会話が主なので長め)
_AGENT_CONV_SILENCE = 1.5     # N秒沈黙で応答(conversation — 発話断片をまとめる)
_INTERRUPT_MIN_CHARS = 8      # ファシリテーター割り込みの最小文字数

# --- 並列ドリフト（脱線）検出 ---
_DRIFT_CHECK_INTERVAL = 1     # ドリフトチェックの発話間隔（1=最後の1言でも即評価）
_DRIFT_CHECK_WINDOW = 6       # チェック時に参照する最近の発話数
_DRIFT_WARMUP = 3             # この発話数に達するまで脱線判定しない（開始時の挨拶の猶予）

# --- デッドエア対策（介入不要後の沈黙ブレーカー） ---
_STALL_SILENCE = 7.0          # 介入不要後この秒数沈黙したら一押し
_STALL_COOLDOWN = 30.0        # 一押しの最小間隔（ループ防止）

# --- エコー防止 ---
_ECHO_COOLDOWN = 2.0          # AI発話終了後のエコーウィンドウ秒数（agent/partner共通）
# 相槌判定: 相槌パターンに一致する発話ではPartnerを止めない
_BACKCHANNEL_RE = re.compile(
    r'^[\s、。,.!?！？]*'
    r'('
    r'うん|ふん|ふーん|へー|ほー|おー|あー|えー'
    r'|はい|ええ|そう|そっか|そうだね|そうですね|そうですか'
    r'|なるほど|確かに|分かる|わかる|分かります|わかりました'
    r'|了解|オッケー|OK'
    r')'
    r'[\s、。,.!?！？うんはいええそっかなるほど確かに]*$',
    re.IGNORECASE,
)
AGENT_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]

_PROMPT_DEBATE_PARTNER = """\
あなたは会議の参加者です。もう一人の参加者（人間）と議題について議論してください。

ルール:
- 自然な日本語で話してください
- 自分の意見を持ち、根拠を示してください
- 相手の意見に同意する場合も反論する場合も、理由を述べてください
- 1回の発言は15秒以内に収まる長さにしてください
- ファシリテーターが介入したら、その指摘を受け止めて議論に反映してください
- 相手が雑談や別の話題を振ってきたら、自然に付き合ってください。\
議題に無理に戻す必要はありません。人間同士の会話のように柔軟に対応してください"""


def fmt_ts(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"
