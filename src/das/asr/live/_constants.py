"""リアルタイム議事録モジュール定数."""
from __future__ import annotations

import os
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

出力はJSONのみ（説明不要）。新しい論点がなければ topics を空配列にしてください。
形式: {{"topics": [{{"topic": "論点の短い要約", "speaker": "発話者名"}}]}}"""

_DRIFT_PROMPT = """\
会議の基準議題・現在の論点・直近の発話を比較し、ファシリテーターが介入すべき
明確な脱線かどうかを判定してください。

## 基準議題
{agenda}

## 現在の論点
{topics}

## 直近の発話
{utterances}

## 判定基準
- 基準議題が明示されている場合だけ、それを「戻るべき軸」として強く扱う
- 基準議題が未指定の場合、現在の論点は会話の流れを理解する参考であり、
  最初の論点へ戻すための固定基準ではない
- 会話が自然に新しい論点へ移っている、検証・メタ会話に移っている、
  参加者がAIやファシリテーターに言及している → drift=false
- 重要でない短い相槌・断片・一時的な余談 → drift=false
- 明らかに無関係な話題が複数発話続き、会話の目的を損ねている → drift=true
- 迷ったら drift=false（過剰な介入を避ける）

JSON1つのみ出力。形式: {{"drift": true/false, "reason": "10字以内"}}"""

_AGENDA_PROMPT = """\
会議の冒頭の発話から、今日の主な議題（テーマ）を一言で推定してください。
まだ議題が定まっていない・挨拶や雑談のみで判断できない場合は空文字を返してください。

## 冒頭の発話
{utterances}

JSON1つのみ出力。形式: {{"agenda": "短い議題。判断不能なら空文字"}}"""

_PARTICIPATION_PROMPT = """\
会議の参加者の発話量と直近の会話から、発言の少ない人にファシリテーターが
声をかけるべきか判定してください。

## 参加者の発話量（直近）
{participation}

## 直近の会話
{utterances}

## 判定基準
- 明らかに発言が少なく、しばらく黙っている人がいて、声かけが自然なら invite=true
- まだ序盤、偏りが小さい、または本人が聞き役で問題ない場合は invite=false
- 一度に声をかけるのは1人だけ

JSON1つのみ出力。形式: {{"invite": true/false, "speaker": "名前。invite=falseなら空文字", "reason": "短い理由"}}"""

_FACTCHECK_PROMPT = """\
あなたは会議中の事実誤りを最小限だけ補正する補助AIです。
判定対象の発話に、今すぐ短く補足すべき明確な事実誤りがあるか判定してください。

目的:
- 会話の流れを止めず、AIエージェントが介入判断に使う重要な事実だけを補正する
- 参加者の表現・比喩・創作設定・言い回しを添削しない
- 意見・好み・仮説・曖昧な記憶違いには介入しない
- ローカルのキーワード判定ではなく、発話の意味を見て判断する

介入してよい例:
- 定義、計算式、単位、日付、制度名、首都、所属、順序などが高確信で誤っている
- 「世界一高い山」「日本一大きい湖」のような、変動しにくい地理・物理量が明確に誤っている
- 音声認識で助詞や語尾が少し崩れていても、断定の意味が明確なら判定してよい
- その誤りを放置すると、以後の議論や判断が明らかにずれる
- 参照文脈は主語や話題の補完だけに使い、訂正するのは判定対象の発話だけ
- 判定対象の発話内の最重要な誤りだけを短く補正する

介入しない例:
- 小説・演技・ロールプレイ・比喩・キャッチコピーなどの表現上の正確さ
- 「襲いかかる」「凶悪」「見抜いていた」のような演出的な言い回し
- 「こう表現した方がよい」「誤解しにくい」のような文体・語彙の提案
- 「たぶん」「なんでしたっけ」だけで、まだ確定主張になっていない
- 単に数式・統計・専門用語が話題に出ただけで、誤った断定がない
- 評価、好み、解釈、予測、研究上の仮説
- ランキング、人口順位、スポーツ順位、価格、最新人数など、時点で変わりやすい内容
- 最新情報や専門領域で確信が持てない内容
- 医療・法律・安全など高リスク領域で、短い一般訂正だけでは危うい内容

## 参照文脈と判定対象
{utterances}

JSON1つのみ出力。形式:
{{
  "should_correct": true/false,
  "confidence": "high" または "medium" または "low",
  "claim": "誤っている主張。複数なら短く列挙。なければ空文字",
  "correction": "会話でそのまま言える短い補足。複数なら2件までを1文で。なければ空文字",
  "reason": "短い理由"
}}"""

_PROMPT_FACILITATOR = """\
あなたは会議のファシリテーターAIです。
参加者の議論を聞いて、必要な時だけ介入してください。

介入すべき場面:
- 議論が行き詰まった時（新しい視点を提案）
- 重要な論点が見落とされている時
- 議論が脱線した時（元のテーマに戻す提案）
- 合意形成が必要な時（要約して確認）
- 高確信の事実誤りがあり、短く補足しないと議論がずれる時

不必要に発言しないでください。人間の議論を尊重し、
本当に価値ある貢献ができる時だけ簡潔に発言してください。
発言は日本語で、30秒以内に収まる長さにしてください。
最初の論点に固着しないでください。会話の論点が自然に移った場合は、
新しい流れを尊重し、元の話題へ戻す必要はありません。
参加者がファシリテーターやAIに話しかけている場合は、脱線扱いで戻すのではなく、
必要ならその問いに短く答えてください。

介入する場合は、前置きや「（介入）」のような記号を付けず、
本題の発言だけを話してください。
介入が不要だと判断した場合は、「（介入不要）」とだけ返してください。"""

_PROMPT_CONVERSATION = """\
あなたは会議に参加しているAIアシスタントです。
参加者と自然に会話してください。質問されたら必ず答えてください。
簡潔に、日本語で返答してください（15秒以内に収まる長さ）。
会議の文脈を踏まえた上で、役に立つ回答を心がけてください。"""

REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2")


def realtime_url(model: str | None = None) -> str:
    return f"wss://api.openai.com/v1/realtime?model={model or REALTIME_MODEL}"


REALTIME_URL = realtime_url()
AGENT_SPEAKER = "ファシリテーター"   # recordsに使うスピーカーキー
UNSURE_SPEAKER = "?"   # 短い発話で話者を確定できないときのキー（表示は「未確定」）
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
_DRIFT_CHECK_INTERVAL = 3     # ドリフトチェックの発話間隔（短い発話に過敏にならない）
_DRIFT_CHECK_WINDOW = 8       # チェック時に参照する最近の発話数
_DRIFT_WARMUP = 3             # この発話数に達するまで脱線判定しない（開始時の挨拶の猶予）
_INTERVENTION_COOLDOWN = 25.0 # 介入後この秒数は脱線介入を抑制（連発=しつこさの防止）

# --- 冒頭アジェンダ自動検出（人間モードで--topic未指定時, S3） ---
_AGENDA_MIN_UTTS = 4          # この発話数たまったら議題推定を試みる
_AGENDA_WINDOW = 12           # 推定に使う冒頭の発話数
_AGENDA_RETRY_SEC = 10.0      # 推定失敗時の再試行間隔（LLM呼び出しの抑制）

# --- 参加度の声かけ（発言の少ない人を誘う, S4） ---
_INVITE_WARMUP = 8            # この発話数たまるまで声かけしない（序盤の偏りは自然）
_INVITE_CHECK_SEC = 8.0       # 参加度チェックの最小間隔（LLM呼び出しの抑制）
_INVITE_QUIET_RATIO = 0.5     # 公平シェアのこの割合を下回る人がいる時のみLLM判定にかける
_INVITE_SILENCE = 2.0         # 声かけは沈黙(間)がこの秒数続いてから（人間を割り込まない）

# --- 事実誤りの短い補正 ---
_FACTCHECK_WINDOW = 3         # 判定に使う直近発話数
_FACTCHECK_CHECK_SEC = 0.5    # LLM判定の最小間隔（事実誤りは早めに補足）
_FACTCHECK_COOLDOWN = 2.0     # 訂正介入の最小間隔（短い補正なので通常介入より短く）
_FACTCHECK_MIN_SILENCE = 1.2  # 訂正介入でも、参加者の発話が切れる短い間を待つ
_FACTCHECK_MIN_CHARS = 8      # 短すぎる発話は事前除外
_FACTCHECK_MAX_RETRIES = 2    # API/JSON一時失敗時の再試行上限（永久詰まり防止）
_FACTCHECK_PENDING_TTL = 30.0 # キュー内の補正が古くなったら会話の自然さを優先して破棄

# --- デッドエア対策（介入不要後の沈黙ブレーカー） ---
_STALL_SILENCE = 7.0          # 介入不要後この秒数沈黙したら一押し
_STALL_COOLDOWN = 30.0        # 一押しの最小間隔（ループ防止）

# --- エコー防止 ---
_ECHO_COOLDOWN = 2.0          # AI発話終了後のエコーウィンドウ秒数（agent/partner共通）

# --- 積極性プロファイル（人間ファシリテーションの介入頻度, S5） ---
# silence_summarize: 沈黙がこの秒数続いたら要約/整理の介入を検討（None=しない）。
# stall_breaker: 「介入不要」後の沈黙に一押しするか。active以外では黙る判断を尊重。
# cooldown: 脱線介入・声かけの最小間隔（しつこさ防止）。
# 既定は controlled。まずは明確な問題時だけ介入する。
_PROACTIVITY_PROFILES = {
    "controlled": {"silence_summarize": None, "cooldown": 40.0,
                   "drift_confirmations": 2, "stall_breaker": False},  # 明確な問題時のみ
    "standard":   {"silence_summarize": 18.0, "cooldown": 25.0,
                   "drift_confirmations": 2, "stall_breaker": False},
    "active":     {"silence_summarize": 8.0,  "cooldown": 15.0,
                   "drift_confirmations": 1, "stall_breaker": True},
}
_PROACTIVITY_DEFAULT = "controlled"
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
