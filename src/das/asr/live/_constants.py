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

_TRIAGE_PROMPT = """\
あなたは会議の発話を分類する補助AIです。判定対象の発話について、次の2点だけを
判定してください。分類のみを行い、訂正文の生成・助言・要約はしません。
参照文脈は主語や話題の補完だけに使い、判定するのは判定対象の発話だけです。

1. factual_claim: 判定対象に「外部知識と照合できる事実の断定」が含まれるか。
   - true の例: 定義・計算式・単位・日付・制度・地理・所属・順序などの断定
     （「指標Xは分子を分母で割った値です」「国Bの首都は都市Aです」）
   - false の例:
     - 意見・好み・評価（「〜が良いと思う」「〜は微妙」）
     - 質問・確認（「〜でしたっけ？」「〜はどこ？」）
     - 「たぶん」「かもしれない」「らしい」等の不確実・伝聞表現
     - 小説・演技・ロールプレイ・比喩などの創作表現
     - 会議の進め方・話題の宣言などのメタ発話（「〜について話しましょう」）
     - 「これ/それ」等の指示語だけで対象が特定できない断定

2. facilitator_request: 判定対象が、AIファシリテーター（呼称の例:
   ファシリテーター、進行役、AI、AIさん、エーアイ）への明示的な呼びかけと依頼か。
   - 依頼している場合だけ、依頼内容を短く書き出す（例: 「ここまでの整理」
     「論点の確認」「次の進行」）
   - AIやファシリテーターを話題として言及しただけの発話
     （「AIは便利ですね」「AIの導入について話しましょう」）は空文字
   - 呼びかけはあるが依頼が無い発話（「AIさんですね」）も空文字

## 参照文脈と判定対象
{utterances}

JSON1つのみ出力。形式:
{{"factual_claim": true/false, "facilitator_request": "依頼内容。なければ空文字"}}"""

_SUMMARY_VALUE_PROMPT = """\
あなたは会議のファシリテーターAIの「今、整理の一言を挟むべきか」を見極める補助AIです。
直近の発話と論点一覧を読み、短い整理・要約の介入が議論に価値を足すかだけを判定してください。
文案の生成や助言はしません。判定だけを返します。

intervene=true の目安:
- 論点が拡散して噛み合っていない（各自が別の話をしている）
- 同じ主張の繰り返しが続き、前に進んでいない
- 決定すべきことが先送りされ続けている

intervene=false の目安（迷ったら false を選ぶ）:
- 議論が順調に深まっている
- 具体案の詰め・作業に入っている
- 直前に整理・要約が入ったばかり

過剰な介入（仕切りすぎ）は議論の妨げになります。価値が曖昧なら黙る（false）。

## 現在の論点
{topics}

## 直近の発話
{utterances}

JSON1つのみ出力。形式:
{{"intervene": true/false, "focus": "介入するなら焦点を短く。しないなら空文字"}}"""

_PROMPT_FACILITATOR = """\
あなたは会議のファシリテーターAIです。
参加者の議論を聞いて、必要な時だけ介入してください。

介入すべき場面:
- 議論が行き詰まった時（新しい視点を提案）
- 重要な論点が見落とされている時
- 議論が脱線した時（元のテーマに戻す提案）
- 合意形成が必要な時（要約して確認）
- 高確信の事実誤りがあり、短く補足しないと議論がずれる時

あなたが呼ばれるのは、介入すべきだと既に判断された場面だけです。
「話すかどうか」は考えず、与えられた文脈に対して簡潔に一言だけ述べてください。
人間の議論を尊重し、価値を足す最小限の発言に留めてください。
足すべき価値が薄いと感じたら、無理に整理せず一言の相槌程度に留めて構いません。
発言は日本語で、30秒以内に収まる長さにしてください。
最初の論点に固着しないでください。会話の論点が自然に移った場合は、
新しい流れを尊重し、元の話題へ戻す必要はありません。
参加者がファシリテーターやAIに話しかけている場合は、脱線扱いで戻すのではなく、
必要ならその問いに短く答えてください。

前置きや「（介入）」のような記号は付けず、本題の発言だけを短く話してください。"""

_PROMPT_CONVERSATION = """\
あなたは会議に参加しているAIアシスタントです。
参加者と自然に会話してください。質問されたら必ず答えてください。
簡潔に、日本語で返答してください（15秒以内に収まる長さ）。
会議の文脈を踏まえた上で、役に立つ回答を心がけてください。"""

# 2026-07: gpt-realtime-2.1 に更新 (割り込み挙動・無音/ノイズ処理の改善)。
# 問題があれば環境変数 OPENAI_REALTIME_MODEL=gpt-realtime-2 で即ロールバック可。
REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")


def realtime_url(model: str | None = None) -> str:
    return f"wss://api.openai.com/v1/realtime?model={model or REALTIME_MODEL}"


AGENT_SPEAKER = "ファシリテーター"   # recordsに使うスピーカーキー
UNSURE_SPEAKER = "?"   # 短い発話で話者を確定できないときのキー（表示は「未確定」）
# pyannote Live-1 のようにセッション序盤でラベルが揺れる（一人の発話が複数の
# 生ラベルに分裂する）providerに対して、新しい外部diarizationラベルを即座に
# 「参加者」として恒久登録しない猶予（ヒステリシス）。同一ラベルの累積発話が
# この秒数に達するまでは @diar:N を新規発行せず UNSURE_SPEAKER（未確定）に
# 留める。SessionState.key_for_diarization_speaker 参照。
PYANNOTE_PARTICIPANT_HYSTERESIS_S = 3.0
# --- pyannote + 声紋照合ハイブリッド構成（クラスタ単位の名前付け, 2026-07-13） ---
# docs/design/pyannote_live1_trial_2026-07-09.md §8.4/§9 参照。pyannoteの生
# クラスタ(SPEAKER_XX)ごとに音声を蓄積し、この秒数に達したら声紋照合を試みる
# （ClusterVoiceNamer, ``_cluster_naming.py``）。閾値未満・照合confidence不足の
# 間は未確定のまま蓄積を続け、再照合のたびに確度が上がる想定。
PYANNOTE_CLUSTER_NAMING_MIN_SEC = 5.0
# クラスタ音声バッファの上限（際限ない保持を防ぐ。古い分から捨てて直近の音声を使う）。
PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC = 20.0
# クラスタ→人物の確定に要求する類似度の下限。match_profile の基準しきい値
# （redimnet 0.42+margin）はクラスタ確定には緩すぎ、低確信の誤確定が一度起きると
# 「確定後は再照合しない」設計により全発話を汚染する（Chiba 0532 実測:
# sim0.54 の誤確定1回で誤帰属37件）。Chiba/YouTube 全6ランの確定イベント
# 10件の実測分布（正: 0.72/0.76/0.77/0.81/0.91/0.92、誤: 0.54/0.58/0.62/0.62）
# の分離帯で校正。redimnet の値（他モデルはスケールが異なるため要再校正。
# handoff_2026-07-14_unregistered_speakers.md §15.9-15.11）。
# 校正の経緯: 0.65（初期校正）→0.70（余白重視）→0.65（確定値, 2026-07-17）。
# 0.70 の実測（2026-07-17_0051）で、正しい確定が 0.66-0.67 にも存在すると判明
# （0.70では正解帯まで削り、未確定44%・対応外15%に悪化）。イベント14件超の
# 分布は 誤: ≤0.62 / 正: ≥0.66 で、分離帯の中央 0.65 を採用。
PYANNOTE_CLUSTER_CONFIRM_MIN_SIM = 0.65
# 不純ラベル発話のクラスタ回収に要求する「声紋の裏付け」の下限類似度（ハイブリッド
# 構成のみ。handoff §18.8）。声紋層が「ラベル不純」で棄権した発話をクラスタ層が
# 回収する経路は、Chiba 12会話の実測で通算正解45%（会話により0-76%）と
# 当てにならず、誤帰属の主経路だった（§18.6）。ただし「その発話自身の声紋1位
# 候補が回収先と一致」している回収は開発5会話で正解37/誤り6と高精度なため、
# 一致かつこの下限以上のときだけ回収を許可する。値は弁別が候補一致そのもので
# 決まりプラトー（0〜0.35でほぼ同成績）のため、無意味に低い類似の偽承認だけを
# 塞ぐ緩い床として 0.25 を採用（sakura01 の最悪条件で誤帰属を9pt追加抑制）。
# redimnet の値（他モデルはスケール要再校正）。
CLUSTER_IMPURE_RECOVERY_ENDORSE_MIN_SIM = 0.25
# --- 鋳造時クラスタリンク（二重帳簿の根治, 2026-07-25。opt-in） ---
# 声紋側が新しい人物Nを鋳造する瞬間に「鋳造したてのプロファイル vs 席を持つ
# クラスタの蓄積声紋」を**対称比較**し、この下限以上（かつ2位と margin 差）なら
# 新しい戸籍を作らず、そのクラスタへ統合する（handoff_2026-07-25_dual_ledger_
# rootcure.md 案B）。同じ人間がクラスタ帳簿(@diar:N)と声紋帳簿(人物N)に二重に
# 載り、統一席ルールの下で席を食い潰して実在者を締め出す問題への対処。
# 校正: 記録16本の鋳造36件を反実仮想採点（同コミット群の eval/phase0_*）。
#   - 鋳造36件中28件が「既に席を持つ人物の二重登録」＝二重帳簿は常態
#   - 同一人物の対称類似 0.50-0.88 / 別人 0.07-0.52。argmax で th=0.50 のとき
#     正25/誤2/取り逃し0（実装式の演算でも 正24/誤1 とほぼ同等＝移植可能）
#   - 0.55 は誤ゼロだが取り逃し4件、0.60 以上は取り逃しが増えるだけ
# **判定は鋳造の瞬間の1回きりに限ること**。蓄積が伸びるたびに再判定する案
# （案B遅延）は時系列シミュレーションで別人の類似が 0.53-0.69 まで上がり
# 分離が消える（短い断片クラスタは誰にでも似る／バッファの直近20秒が別人の
# 声になる）。これは §15.12 で分離閾値なしと実測し §18.9 で削除したクラスタ間
# 名寄せと同じ失敗モードで、繰り返し判定は「最悪の読みがいつか当たる」露出を
# 作る。一発勝負が成立するのは、鋳造時のプロファイルが純度検査を通った登録
# 材料から作りたてで、比較が1回きりだから。
# redimnet の値（他モデルはスケールが異なるため要再校正）。
PYANNOTE_CLUSTER_MINT_LINK_MIN_SIM = 0.50
# 鋳造時クラスタリンクの2位との差（誤リンクは1つのクラスタが複数人に中程度に
# 似るときに起きるため、1位が2位を明確に上回ることを要求する）。既存の
# match_profile と同じ 0.05（実測でも 0.05 を課して 正24/誤1 と成績不変）。
PYANNOTE_CLUSTER_MINT_LINK_MARGIN = 0.05
# --- 席落ち発話の割当て（クラスタ分裂の回収, 2026-07-26。handoff §27） ---
# pyannote が同じ人を複数クラスタに割ると、割れた側は席上限で落ちて未確定になる。
# 実測（Chiba 9ラン）では実質発話の 11.8% がこれで、落ちたキーは**全て** @diar:N
# ＝既に席を持つ人の分裂だった。そこで「席上限で落ちる発話」に限り、席を持つ人の
# 実音声と比べて最も似た1人へその発話だけ寄せる。
# これは §15.12/§18.9 で削除したクラスタ間名寄せとは別物である。あちらは
# 「既存の誰かか、新しい人か」という開集合の判定で、分離できる閾値が存在しない
# （§27.7 で別人が 0.89 に達することを再確認した）。こちらは席上限に達していて
# 新しい参加者が入れないことが確定した状態＝閉集合の割当てで、しかも確定を
# 書かず1発話限り。したがって類似度の下限は課さない（課しても成績はほぼ変わらず、
# 実測で「しきい値は効いていない」ことが分かっている）。
# 参照に人物プロファイルではなく席の実音声を使うのは、プロファイル（短窓の登録
# サンプル由来）相手だと同じ分裂クラスタが確定線に届かないのに、席の実音声相手
# なら 0.78-0.92 出るため（§27.4）。
# 席あたりの参照音声の秒数。貯まったら凍結する（席には誤帰属も混ざるので貯め
# 続けると参照が汚れる。埋め込みの再計算コストがライブの遅延に効くのも理由）。
SEAT_AUDIO_REF_SEC = 30.0
# 参照がこの秒数に育っていない席は候補から外す（その席へは寄せない）。
# 校正（Chiba 9ラン・因果的に過去の音声だけで参照を作った実測。handoff §27.8）:
#   下限 0秒 → 適用157件・1位正解63%・正解 +7.5pt / 誤帰属 +4.2pt
#   下限 3秒 → 適用153件・1位正解65%・正解 +7.4pt / 誤帰属 +3.9pt
#   下限12秒 → 適用 93件・1位正解71%・正解 +4.8pt / 誤帰属 +2.2pt
# 0〜8秒はプラトーで**値は効いていない**（精度が上がるのは12秒以降だが、
# 適用が6割に減って正解の取り分が半分になる）。したがってこの定数は成績の
# チューニングつまみではなく、「数秒しか聞いていない席には寄せない」という
# 下限として置く。プラトーの内側なので過学習の心配も無い。
SEAT_AUDIO_MIN_REF_SEC = 3.0
# 重複発話（同時に複数の生クラスタが閾値以上を占める）区間は、声が混ざり声紋が
# あてにならないため安全側で未確定にする。この比率以上を占める話者が2人以上いれば
# 重複発話とみなす。
PYANNOTE_CLUSTER_OVERLAP_MIN_RATIO = 0.2
# =====================================================================
# タイミング定数（介入・割り込みロジック）
# ---------------------------------------------------------------------
# 関係の概要:
#   - ファシリテーターの通常トリガーは「N発話到達」または「一定時間の沈黙」。
#     debateモードはPartner会話が主なので沈黙閾値を長めにする。
#   - 並列ドリフトチェッカーは、ウォームアップ後、INTERVAL発話ごとに
#     直近WINDOW発話を見て脱線判定する。
#   - エコーウィンドウ = AI発話終了後 ECHO_COOLDOWN 秒。この間はトリガー抑止＆
#     テキスト類似エコー除去を適用する。
# =====================================================================

# --- ファシリテーターの通常トリガー ---
_AGENT_TRIGGER = 10           # N発話ごとに応答検討(facilitator)
_AGENT_DEBATE_SILENCE = 15.0  # N秒沈黙で応答検討(debate — Partner会話が主なので長め)
_AGENT_CONV_SILENCE = 1.5     # N秒沈黙で応答(conversation — 発話断片をまとめる)
_INTERRUPT_MIN_CHARS = 8      # ファシリテーター割り込みの最小文字数

# --- 並列ドリフト（脱線）検出 ---
_DRIFT_CHECK_INTERVAL = 3     # ドリフトチェックの発話間隔（短い発話に過敏にならない）
_DRIFT_CHECK_WINDOW = 8       # チェック時に参照する最近の発話数
_DRIFT_WARMUP = 3             # この発話数に達するまで脱線判定しない（開始時の挨拶の猶予）
_INTERVENTION_COOLDOWN = 25.0 # 介入後この秒数は脱線介入を抑制（連発=しつこさの防止）
_DRIFT_PENDING_TTL = 30.0     # 保留中の脱線候補の寿命。確認待ちのまま古くなったら破棄
                              # （会話が自然に本題へ戻ったのに候補が残り続けるのを防ぐ）

# --- 同一内容介入の再発火抑止（2026-07-22 実利用での再発報告に対応） ---
# 時間クールダウンは「間隔」しか見ないため、会話が停滞すると時間を置いて
# 同じ内容の介入（同じ脱線理由・同じ整理焦点）が再発火し、表示が繰り返される。
# brief が内容そのものである種別（drift/summarize）に限り、直近の同種介入と
# 実質同一の候補はこの窓の間は採らない。窓は「会議が同じ論点に留まりうる
# 時間」の上限の目安として 10 分。
_INTERVENTION_CONTENT_DEDUP_SEC = 600.0
# 実質同一の判定床。brief は LLM が毎回生成するため文言が揺れ、完全一致では
# 取り逃す。空白除去後の SequenceMatcher 類似がこの値以上なら同一とみなす。
_INTERVENTION_CONTENT_DEDUP_SIM = 0.6

# --- 整理介入の価値判定（C3, count の無条件介入を置換） ---
_STRUCTURING_WINDOW = 12      # 価値判定に渡す直近の発話数

# --- 冒頭アジェンダ自動検出（人間モードで--topic未指定時, S3） ---
_AGENDA_MIN_UTTS = 4          # この発話数たまったら議題推定を試みる
_AGENDA_WINDOW = 12           # 推定に使う冒頭の発話数
_AGENDA_RETRY_SEC = 10.0      # 推定失敗時の再試行間隔（LLM呼び出しの抑制）

# --- 参加度の声かけ（発言の少ない人を誘う, S4） ---
_INVITE_WARMUP = 8            # この発話数たまるまで声かけしない（序盤の偏りは自然）
_INVITE_CHECK_SEC = 8.0       # 参加度チェックの最小間隔（LLM呼び出しの抑制）
_INVITE_QUIET_RATIO = 0.5     # 公平シェアのこの割合を下回る人がいる時のみLLM判定にかける
_INVITE_SILENCE = 2.0         # 声かけは沈黙(間)がこの秒数続いてから（人間を割り込まない）

# --- 発話の表層分類（fact候補・ファシリテーター呼びかけ, H6/M2） ---
_TRIAGE_MIN_CHARS = 4         # これ未満の発話は分類せず「候補なし」扱い（コスト0のゲート）
_TRIAGE_MAX_RETRIES = 2       # API/JSON一時失敗時の再試行上限（永久詰まり防止）
_TRIAGE_CONTEXT_WINDOW = 3    # 分類時に参照する直前の発話数（指示語・省略の補完用）
_TRIAGE_BACKLOG_MAX = 8       # 1tickで連続処理する上限。遅延を有界にする
#                              （8件×2秒≈最悪16秒の追いつき時間）。これを超える
#                              古いバックログは分類せず負注釈でスキップする

# --- 事実誤りの短い補正 ---
_FACTCHECK_CHECK_SEC = 0.5    # LLM判定の最小間隔（事実誤りは早めに補足）
_FACTCHECK_COOLDOWN = 2.0     # 訂正介入の最小間隔（短い補正なので通常介入より短く）
_FACTCHECK_MIN_CHARS = 8      # 短すぎる発話は事前除外
_FACTCHECK_MAX_RETRIES = 2    # API/JSON一時失敗時の再試行上限（永久詰まり防止）
_FACTCHECK_PENDING_TTL = 30.0 # キュー内の補正が古くなったら会話の自然さを優先して破棄

# --- 介入タイミング ---
_INTERVENTION_PAUSE_FACT = 0.9   # 事実補正: 鮮度優先。ただし発話には被せない
_INTERVENTION_PAUSE_DRIFT = 1.8  # 脱線: 会話の自律的な復帰を少し待つ
_INTERVENTION_PAUSE_RETRY = 2.4  # 再送: しつこさを避け、十分な間がある時だけ
_INTERVENTION_PAUSE_COUNT = 1.5  # 発話数整理: 参加者の連続発話を遮らない
_INTERVENTION_PAUSE_MANUAL = 1.0  # 手動呼び出し: 発話には被せないが drift/invite より早く反応

# --- 手動呼び出し（ファシリテーターを明示的に呼ぶ, Phase1） ---
_MANUAL_CALL_COOLDOWN = 5.0      # 同種の連打防止（global cooldown の影響は受けない）
_MANUAL_CALL_TTL = 30.0          # 古すぎる手動呼び出しは破棄
_MANUAL_CALL_MAX_CHARS = 100     # 依頼文の最大長（超過分は切り詰め）
# 事前登録の品質ゲート（P2-5）。無音を除いた実効音声長がこの秒数未満なら reject。
_ENROLL_MIN_VOICED_SEC = 2.0
# 音声呼びかけ検出時の即時アック音（H）。呼びかけが「聞こえた」を短いチャイムで
# 伝え、言い直し（二重呼び出し）を減らす。実験条件で音を消したい時は False に。
_ACK_CHIME_ENABLED = True

# --- エコー防止 ---
_ECHO_COOLDOWN = 2.0          # AI発話終了後のエコーウィンドウ秒数（agent/partner共通）

# --- フロア判定（発話被り防止, F3/M6） ---
# アクティブな partial（誰かの発話が今まさに転写されている）間はフロアを占有中と
# みなし、介入のpause判定を保守側に倒す。partial がこの秒数以上変化していなければ
# stale とみなして無視する（partial がクリアされずに固着した場合の保険）。
_PARTIAL_FLOOR_MAX_AGE = 10.0

# --- 積極性プロファイル（人間ファシリテーションの介入頻度, S5） ---
# silence_summarize: 沈黙がこの秒数続いたら要約/整理の介入を検討（None=しない）。
# cooldown: 脱線介入・声かけの最小間隔（しつこさ防止）。
# drift_confirmations: 脱線を採るまでに必要な連続検出回数。
# 既定は standard。デモや通常利用では、沈黙時の短い整理も許可する。
# 注: 旧 stall_breaker（「介入不要」後のデッドエア一押し）は Phase3 で廃止した。
# Speaker から「介入不要」判断を外したため、その履歴に依存する一押しは行わない。
_PROACTIVITY_PROFILES = {
    "controlled": {"silence_summarize": None, "cooldown": 40.0,
                   "drift_confirmations": 2},  # 明確な問題時のみ
    "standard":   {"silence_summarize": 18.0, "cooldown": 25.0,
                   "drift_confirmations": 2},
    "active":     {"silence_summarize": 8.0,  "cooldown": 15.0,
                   "drift_confirmations": 1},
}
_PROACTIVITY_DEFAULT = "standard"
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

# --- 遡及訂正（序盤の帰属を、席の参照が育ってから貼り直す。handoff §28） ---
# 誤りはセッション序盤に極端に偏る（実測: 0-1分 正解29% / 1-2分 69% /
# 5-10分 90%）。システムは収束しており悪いのは立ち上がりだけなので、参照が
# 育った時点で序盤の発話を決め直す。
# 校正: 2分・5分は §28.2 の実測点（79.2%→84.5%→89.5%）。以降の間隔は
# §28.5 で予定表を比べて決めた（本番クラスをそのまま駆動した実測）:
#   5分ごと 85.2% / **2分ごと 89.9%** / 1分ごと 89.9% / 発話ごと 89.9%
# 2分でプラトーに達し、それ以上細かくしても上がらない。貼り直し自体は
# 保存済みの声紋との内積だけで埋め込みの計算が要らない＝ほぼ無料なので、
# 間隔を詰めない理由は計算量ではなく「表示が頻繁に書き換わる」ことだけ。
# したがって**プラトーの入口**である2分を採る（10分の会話で7回）。
# 初回を60秒にしているのは表示のため。貼り直しは表示ラベルの詰め直しも
# 伴う（§28.6）ので、それまでは「参加者が1人なのにBから始まる」状態が
# 見えたままになる。§28.5 の測定で「1分ごと」も 89.9% とプラトー内に
# あることが確認できているため、精度を落とさずに醜い窓を半分にできる。
RETRO_SCHEDULE_SEC = (60.0, 120.0, 300.0)
RETRO_INTERVAL_SEC = 120.0
