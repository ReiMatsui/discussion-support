#!/usr/bin/env python3
"""ファシリテーションAIのテストハーネス（Phase 1: テキストレベル）。

スクリプト化した会議シナリオを流し、AIの介入判断と内容を検証する。
Realtime APIではなくChat APIを使うことで高速・低コストに反復できる。

使い方:
  # 全シナリオを実行
  python -m tests.test_facilitation_harness

  # 特定シナリオだけ
  python -m tests.test_facilitation_harness --scenario stalled

  # カスタムプロンプトを試す
  python -m tests.test_facilitation_harness --prompt-file my_prompt.txt

必要:
  export OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

# --- プロジェクトルートをパスに追加 ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  シナリオ定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Utterance:
    speaker: str
    text: str

@dataclass
class TriggerPoint:
    """この位置で介入判断を要求する."""
    after_utterance: int          # 何番目の発話の後か（0-indexed）
    expected: Literal["intervene", "skip", "any"]  # 期待される判断
    context: str = ""             # この時点で期待される状況の説明

@dataclass
class Scenario:
    name: str
    description: str
    utterances: list[Utterance]
    triggers: list[TriggerPoint]
    tags: list[str] = field(default_factory=list)


# --- 組み込みシナリオ ---

SCENARIOS: dict[str, Scenario] = {}

def _register(s: Scenario):
    SCENARIOS[s.name] = s

# 1. 議論停滞: 同じ話題を繰り返し、新しい論点が出ない
_register(Scenario(
    name="stalled",
    description="議論が停滞し同じ論点を繰り返している。AIは新しい視点を提案すべき。",
    tags=["停滞", "介入期待"],
    utterances=[
        Utterance("松井", "やっぱりコストが一番の問題だと思います。"),
        Utterance("田中", "そうですね、コストは確かに高いですよね。"),
        Utterance("松井", "コスト削減をなんとかしないと先に進めないですよね。"),
        Utterance("田中", "ええ、コストの問題は大きいです。"),
        Utterance("松井", "とにかくコストをどうするかですよね。"),
        Utterance("田中", "コスト面での対策が必要ですね。"),
        Utterance("松井", "何かコストを下げる方法はないですかね。"),
        Utterance("田中", "うーん、コストか…難しいですね。"),
    ],
    triggers=[
        TriggerPoint(after_utterance=3, expected="skip",
                     context="まだ4発話。停滞判定には早い"),
        TriggerPoint(after_utterance=7, expected="intervene",
                     context="8発話で同じ話題を繰り返し。新しい視点の提案を期待"),
    ],
))

# 2. 偏り: 賛成意見ばかりで反論がない
_register(Scenario(
    name="biased",
    description="全員が賛成意見ばかり。反対意見やリスクが見落とされている。",
    tags=["偏り", "介入期待"],
    utterances=[
        Utterance("松井", "新しいAIツールを導入しましょう。業務効率が上がるはずです。"),
        Utterance("田中", "賛成です。最近のAIはすごく進化してますからね。"),
        Utterance("佐藤", "私も賛成です。競合他社も導入してますし。"),
        Utterance("松井", "じゃあ早速来月から導入の準備を始めましょう。"),
        Utterance("田中", "いいですね。ベンダーに見積もりを取りましょう。"),
        Utterance("佐藤", "予算は問題ないと思います。効果を考えれば安いものです。"),
    ],
    triggers=[
        TriggerPoint(after_utterance=5, expected="intervene",
                     context="全員賛成でリスク・コスト・運用負荷の検討がない。バランスの指摘を期待"),
    ],
))

# 3. 脱線: 本題から逸れていく
_register(Scenario(
    name="derailed",
    description="本題（プロジェクト計画）から雑談に脱線。AIは元のテーマに戻すべき。",
    tags=["脱線", "介入期待"],
    utterances=[
        Utterance("松井", "では次のスプリントの計画を決めましょう。"),
        Utterance("田中", "はい、まずバックログの優先順位を…あ、そういえば昨日のサッカー見ました？"),
        Utterance("佐藤", "見ましたよ！すごい試合でしたね。"),
        Utterance("田中", "後半のゴールがすごかったですよね。"),
        Utterance("松井", "確かに。でも審判の判定はどうかと思いましたけど。"),
        Utterance("佐藤", "VAR導入してからああいう判定増えましたよね。"),
        Utterance("田中", "スポーツとテクノロジーの関係って面白いですよね。"),
    ],
    triggers=[
        TriggerPoint(after_utterance=3, expected="any",
                     context="脱線し始めたが、まだ自然に戻る可能性もある"),
        TriggerPoint(after_utterance=6, expected="intervene",
                     context="完全に脱線。スプリント計画に戻す提案を期待"),
    ],
))

# 4. 正常な議論: AIは黙るべき
_register(Scenario(
    name="healthy",
    description="活発で建設的な議論。AIの介入は不要。",
    tags=["正常", "スキップ期待"],
    utterances=[
        Utterance("松井", "認証方式はOAuth2.0にしたいと思います。"),
        Utterance("田中", "いいと思います。ただ、トークンのリフレッシュ戦略はどうしますか？"),
        Utterance("佐藤", "サイレントリフレッシュがUX的にはベストですが、セキュリティ面が気になります。"),
        Utterance("松井", "確かに。リフレッシュトークンのローテーションを入れれば緩和できるかと。"),
        Utterance("田中", "それならアクセストークンの有効期限も短めにできますね。15分くらい？"),
        Utterance("佐藤", "15分でいいと思います。オフラインアクセスが必要な場合は別途検討しましょう。"),
        Utterance("松井", "では認証はOAuth2.0 + トークンローテーション + 15分有効期限で進めましょう。"),
    ],
    triggers=[
        TriggerPoint(after_utterance=6, expected="skip",
                     context="建設的に議論が進み合意形成もできている。介入不要"),
    ],
))

# 5. 合意形成が必要: 意見が分かれたまま先に進もうとしている
_register(Scenario(
    name="consensus_needed",
    description="意見が割れたまま結論を出そうとしている。整理・確認が必要。",
    tags=["合意形成", "介入期待"],
    utterances=[
        Utterance("松井", "リリースは来週金曜日にしましょう。"),
        Utterance("田中", "来週は早すぎます。テストが間に合わないかもしれません。"),
        Utterance("佐藤", "でも顧客への約束があるので遅らせられないです。"),
        Utterance("田中", "品質を犠牲にしてまでリリースすべきじゃないと思います。"),
        Utterance("松井", "じゃあもう来週金曜で決定ということで。次の議題に移りましょう。"),
    ],
    triggers=[
        TriggerPoint(after_utterance=4, expected="intervene",
                     context="田中の懸念が解決されないまま松井が強引に決定。整理を期待"),
    ],
))

# 6. 一人が独占: 特定の人だけが話し続けている
_register(Scenario(
    name="monopolized",
    description="一人が発言を独占し、他の参加者が意見を言えていない。",
    tags=["偏り", "介入期待"],
    utterances=[
        Utterance("松井", "このプロジェクトは私が考えた方法で進めるべきです。"),
        Utterance("松井", "まずデータベースの設計から始めます。PostgreSQLを使います。"),
        Utterance("松井", "APIはRESTで、フロントエンドはReactです。"),
        Utterance("松井", "デプロイはAWSのECSを使います。"),
        Utterance("松井", "テストはJestとPytestで十分でしょう。"),
        Utterance("松井", "スケジュールは3ヶ月で、マイルストーンは月次で設定します。"),
        Utterance("田中", "あの…"),
        Utterance("松井", "CI/CDはGitHub Actionsで組みます。"),
    ],
    triggers=[
        TriggerPoint(after_utterance=5, expected="intervene",
                     context="松井が6発話連続。田中・佐藤に意見を聞くべき"),
        TriggerPoint(after_utterance=7, expected="intervene",
                     context="田中が発言しようとしたが松井が遮った。他の参加者への配慮を促すべき"),
    ],
))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ファシリテーション プロンプト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_SYSTEM_PROMPT = """\
あなたは会議のファシリテーターAIです。
参加者の議論を聞いて、**本当に必要な時だけ**介入してください。

## 最重要原則: 黙ることがデフォルト
- 迷ったら黙る（skip）。介入は確信がある時だけ。
- 議論が前に進んでいるなら、たとえ改善の余地があっても黙る。
- 「こうすればもっと良くなる」程度では介入しない。「このままでは問題が起きる」レベルで介入する。
- 参加者が自分たちで論点に気づける余地があるなら、まず待つ。

## 介入すべき場面（これらが**明確に**起きている時だけ）
1. **停滞**: 同じ論点を3回以上繰り返し、新しい視点が全く出ていない（序盤の繰り返しは自然なので待つ）
2. **重大な見落とし**: リスクや反対意見が**完全に**無視されたまま結論に向かっている
3. **脱線**: 本題と無関係な話が**3発話以上**続いている
4. **強引な合意**: 反対意見が未解決のまま議長が結論を押し通そうとしている
5. **発言独占**: 1人が**5発話以上**連続し、他の参加者が発言できていない

## 介入しないべき場面
- 議論が建設的に進んでいる（たとえ完璧でなくても）
- まだ序盤で議論が温まっていない（最低5発話は見守る）
- 参加者が自分たちで軌道修正できそうな時
- 「あったら良い」程度の補足情報

## 発言ルール
- 日本語で、30秒以内に収まる長さ（100文字程度）
- 指示ではなく提案の形（「〜しませんか？」「〜はいかがでしょう？」）

## 応答フォーマット
以下のJSON形式で応答してください:
{
  "decision": "intervene" または "skip",
  "reason": "判断の根拠（1文）",
  "utterance": "介入する場合の発言内容（skipの場合は空文字）",
  "type": "介入タイプ: new_perspective / balance / redirect / summarize / include / skip"
}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  テスト実行エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TriggerResult:
    trigger: TriggerPoint
    response: dict          # パースされたJSON応答
    raw: str                # 生のAPI応答
    match: bool             # expected と一致したか
    latency_ms: float


@dataclass
class ScenarioResult:
    scenario: Scenario
    trigger_results: list[TriggerResult]
    total_tokens: int = 0

    @property
    def pass_rate(self) -> float:
        evaluated = [r for r in self.trigger_results if r.trigger.expected != "any"]
        if not evaluated:
            return 1.0
        return sum(1 for r in evaluated if r.match) / len(evaluated)


def _call_chat_api(system_prompt: str, conversation: str, model: str) -> tuple[str, int]:
    """OpenAI Chat APIを呼び出し、応答テキストとトークン数を返す."""
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens


def _format_conversation(utterances: list[Utterance]) -> str:
    """発話リストを会話テキストに整形."""
    lines = []
    for i, u in enumerate(utterances):
        lines.append(f"[発話{i+1}] {u.speaker}: {u.text}")
    return "\n".join(lines)


def run_scenario(
    scenario: Scenario,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> ScenarioResult:
    """1つのシナリオを実行し結果を返す."""
    results: list[TriggerResult] = []
    total_tokens = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"📋 シナリオ: {scenario.name}")
        print(f"   {scenario.description}")
        print(f"{'='*60}")

    for trigger in scenario.triggers:
        # トリガーポイントまでの発話を切り出し
        context_utterances = scenario.utterances[:trigger.after_utterance + 1]
        conv_text = _format_conversation(context_utterances)

        if verbose:
            print(f"\n--- トリガー: 発話{trigger.after_utterance + 1}の後 ---")
            print(f"期待: {trigger.expected} ({trigger.context})")

        t0 = time.monotonic()
        raw, tokens = _call_chat_api(system_prompt, conv_text, model)
        latency = (time.monotonic() - t0) * 1000
        total_tokens += tokens

        # JSON応答をパース
        try:
            response = json.loads(raw)
        except json.JSONDecodeError:
            response = {"decision": "error", "reason": "JSONパース失敗", "utterance": "", "type": "error"}

        decision = response.get("decision", "error")
        match = (trigger.expected == "any"
                 or (trigger.expected == "intervene" and decision == "intervene")
                 or (trigger.expected == "skip" and decision == "skip"))

        result = TriggerResult(
            trigger=trigger,
            response=response,
            raw=raw,
            match=match,
            latency_ms=latency,
        )
        results.append(result)

        if verbose:
            icon = "✅" if match else "❌"
            print(f"{icon} 判定: {decision} ({response.get('type', '?')})")
            print(f"   理由: {response.get('reason', '?')}")
            if decision == "intervene":
                print(f"   発言: {response.get('utterance', '?')}")
            print(f"   ({latency:.0f}ms, {tokens}tok)")

    return ScenarioResult(scenario=scenario, trigger_results=results, total_tokens=total_tokens)


def run_all(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = "gpt-4o-mini",
    scenario_filter: str | None = None,
    verbose: bool = True,
) -> list[ScenarioResult]:
    """全シナリオ（またはフィルタ指定）を実行."""
    results = []
    scenarios = SCENARIOS
    if scenario_filter:
        scenarios = {k: v for k, v in SCENARIOS.items() if scenario_filter in k}
        if not scenarios:
            print(f"❌ シナリオ '{scenario_filter}' が見つかりません。")
            print(f"   利用可能: {', '.join(SCENARIOS.keys())}")
            return []

    for scenario in scenarios.values():
        result = run_scenario(scenario, system_prompt, model, verbose)
        results.append(result)

    # サマリ
    if verbose and len(results) > 1:
        print(f"\n{'='*60}")
        print("📊 サマリ")
        print(f"{'='*60}")
        total_tokens = 0
        for r in results:
            icon = "✅" if r.pass_rate == 1.0 else "⚠️" if r.pass_rate >= 0.5 else "❌"
            print(f"  {icon} {r.scenario.name}: {r.pass_rate:.0%} "
                  f"({sum(1 for tr in r.trigger_results if tr.match)}/{len(r.trigger_results)})")
            total_tokens += r.total_tokens
        overall = sum(r.pass_rate for r in results) / len(results)
        print(f"\n  全体: {overall:.0%} | 合計トークン: {total_tokens}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(description="ファシリテーションAI テストハーネス")
    ap.add_argument("--scenario", "-s", help="実行するシナリオ名（部分一致）")
    ap.add_argument("--model", "-m", default="gpt-4o-mini", help="使用するモデル")
    ap.add_argument("--prompt-file", "-p", help="カスタムプロンプトファイル")
    ap.add_argument("--list", "-l", action="store_true", help="シナリオ一覧を表示")
    ap.add_argument("--json", action="store_true", help="結果をJSON出力")
    args = ap.parse_args()

    if args.list:
        for name, s in SCENARIOS.items():
            print(f"  {name:20s} [{', '.join(s.tags)}] {s.description}")
        return

    from das.asr.soniox_live import load_env
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY が設定されていません。(.env または環境変数)", file=sys.stderr)
        sys.exit(1)

    prompt = DEFAULT_SYSTEM_PROMPT
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()
        print(f"# カスタムプロンプトを読み込みました: {args.prompt_file}")

    results = run_all(prompt, args.model, args.scenario, verbose=not args.json)

    if args.json:
        out = []
        for r in results:
            out.append({
                "scenario": r.scenario.name,
                "pass_rate": r.pass_rate,
                "total_tokens": r.total_tokens,
                "triggers": [
                    {
                        "after_utterance": tr.trigger.after_utterance,
                        "expected": tr.trigger.expected,
                        "decision": tr.response.get("decision"),
                        "type": tr.response.get("type"),
                        "reason": tr.response.get("reason"),
                        "utterance": tr.response.get("utterance"),
                        "match": tr.match,
                        "latency_ms": tr.latency_ms,
                    }
                    for tr in r.trigger_results
                ],
            })
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
