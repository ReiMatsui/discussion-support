# 音声 AI ファシリテーター介入設計・実装計画

最終更新: 2026-06-25  
ステータス: MVP 方針決定・詳細設計中  
対象: `src/das/asr/live/` の音声 AI ファシリテーター  

## 0. この文書の位置づけ

この文書を、音声 AI ファシリテーターの介入判断に関する **設計の正本** とする。
今後、介入原則、検出対象、閾値、ファイル構造、実装順序を変更するときは、
コード変更に先立って、または同じ変更単位でこの文書を更新する。

既存文書との役割分担:

- `phase3_plan.md`: Realtime Agent の並行性、割り込み、状態管理の構造改善
- `human_facilitation_plan.md`: 現行の脱線検出・参加度声かけ機能の導入計画
- `presentation_policy.md`: 議論グラフから参加者へ何を提示するか
- **本書**: 音声 AI が「何を観察し、いつ、どの強さで、どう介入するか」

本書は、`deep-research-report.md` で整理された次の原則を実装可能な形に落とす。

1. 先に観察し、単発ではなく問題の継続性を見る
2. 安全上の問題を除き、できるだけ遅く介入する
3. 有効である限り最も弱い介入から始める
4. 介入後に改善を確認し、改善しない場合だけ強める
5. 介入後は会話の主導権を参加者へ返す

---

## 1. スコープ

### 対象

- 人間 2〜3 名の同期的な対面会議
- Soniox で確定した発話を入力とする音声ファシリテーション
- OpenAI Realtime API による短い音声介入
- 発話内容、話者、発話時間、沈黙、介入履歴を使った判断
- ライブ UI での状態表示、設定、診断ログ

### 今回の非対象

- 議論グラフを必須とする介入
- Web 検索や外部文書を使った内容提供
- AI 会話パートナー自体の応答方針
- 大規模オンライン掲示板のモデレーション
- 医療、法務、緊急対応など専門判断を伴う安全判定
- 完全自動の会議目的・決定期限・組織ルール推定

議論グラフとの統合は将来可能にするが、音声ファシリテーターの基本動作は
`python -m das.asr.live --agent` 単体で成立させる。

---

## 2. MVP で決定した方針

### 2.1 役割

MVP の音声 AI ファシリテーターは、**中立的な司会・進行支援役**に限定する。

行うこと:

- 発話の少ない参加者へ、任意回答の形で意見を求める
- 議題からの継続的な脱線を短く指摘する
- 参加者から明示的に求められた場合に、議論を短く整理する
- 停滞や反復の兆候を観測し、検証用ログに残す

行わないこと:

- AI 自身の意見、賛否、結論を述べる
- 新しい論点、外部知識、解決策を自発的に提示する
- 参加者の発言を正しい・間違いと評価する
- 合意や意思決定をAIが代行する
- 発言を強制する

発言を求める場合は、必ず回答を辞退できる語法にする。

```text
良い例:
「田中さん、もしよければ、この点についてどう考えますか？」

避ける例:
「田中さんも意見を言ってください。」
```

### 2.2 対象シナリオ

- 人間 2〜3 名
- 対面での同期会議
- 1つの主要議題を扱う
- AIは参加者ではなく司会役
- 会議時間、組織ルール、専門知識は明示入力されない前提

### 2.3 MVP の診断軸

| 軸 | MVPでの扱い |
|---|---|
| `focus` | 自動介入対象。継続的な脱線を検出する |
| `participation` | 自動介入対象。発話量の偏りを検出する |
| `help_request` | 自動介入対象。明示的な支援要請を検出する |
| `progress` | 診断・ログ対象。停滞の自動介入はMVPでは行わない |

`safety`、`understanding`、`consensus` は将来フェーズとし、MVPの自動介入には
含めない。

### 2.4 MVP の発火条件

| 軸 | 検出条件 | 発火条件 |
|---|---|---|
| `focus` | 直近発話が議題から逸脱している | 2回連続検出 |
| `participation` | 8発話以降、発話量の明確な偏りを検出 | 2回連続判定 |
| `help_request` | 整理・進行支援を明示的に求める | 1回で発火 |
| `progress` | 沈黙、反復、進展不足を検出 | MVPでは発火しない。観測ログのみ |

`focus` について、LLMが高信頼と判断してもMVPでは1回で発火させない。
安全問題を扱わないMVPでは、例外的な即時介入経路を設けない。

### 2.5 MVP の介入レベル

```text
Level 1
  → 介入後3発話を観察
  → 改善しなければLevel 2
  → さらに4発話を観察
  → 改善しなければLevel 3相当として記録
```

| Level | 内容 |
|---|---|
| 1 | 質問・確認。参加者による自己修正を促す |
| 2 | 要約・論点整理・進め方の提案 |
| 3 | ルール確認・一時停止・進め方自体の確認 |

同一問題への自動介入は最大2回なので、MVPで音声として実行するのはLevel 1と
Level 2までとする。Level 2後も改善しない場合はLevel 3相当の状態としてログへ
残すが、3回目の音声介入は行わない。これは介入レベル案と過介入上限を両立させる
ためのMVP制約である。

`participation` はLevel 2でも語調を強めず、個人への再指名ではなく全員への問いへ
切り替える。`help_request` は依頼範囲を超えず、必要なら「さらに整理が必要か」を
確認する。

### 2.6 発話タイミング

- 通常介入は2秒の沈黙を待つ
- MVPでは人間の発話へ割り込まない
- 人間が話し始めたらAIは必ず譲る
- 中断された介入は、問題が継続している場合だけ再試行する
- 一度の介入は20秒以内とする

### 2.7 効果判定

介入後3発話を基本観察窓とする。Level 2後は4発話を観察する。

| 介入 | 改善とみなす状態 |
|---|---|
| `redirect` | 議題または対象論点へ戻る |
| `invite` | 対象者が発言する、または明示的に辞退する |
| `summarize` | 要約への同意・修正、または次の論点への移行が起きる |

### 2.8 過介入の上限

- 全体クールダウン: 20秒
- 同一問題への介入: 最大2回
- 同一人物への連続介入: 禁止
- 1分あたりの介入: 最大2回

上限へ達した場合は自動介入を止め、診断ログだけを継続する。

### 2.9 MVP の評価基準

- 不要な介入率
- 必要な介入の見逃し率
- タイミングの適切性
- 中立性
- しつこさ
- 介入後の改善率
- AIが人間に遮られた割合

---

## 3. 現在の実装

### 3.1 データフロー

```text
マイク
  → Soniox stt-rt-v5
  → 話者ダイアライゼーション
  → 声紋補正・AIエコー除去
  → 確定発話を SessionState.records に保存
  → RealtimeAgent.feed()
  → _run_agent_worker がトリガーを調停
  → RealtimeAgent.trigger()
  → OpenAI gpt-realtime-2
  → ストリーミング音声再生
```

### 3.2 現在のトリガー

| トリガー | 現在の動作 |
|---|---|
| 発話数 | 既定 10 発話ごとに介入要否を Realtime モデルへ問い合わせる |
| 沈黙 | `standard=18秒`、`active=8秒`。`controlled` は沈黙だけでは発火しない |
| 脱線 | 3 発話のウォームアップ後、直近 6 発話を LLM 判定 |
| 参加偏り | 8 発話後、発話時間シェアが公平値の半分未満の候補を LLM 判定 |
| 停滞 | 「介入不要」の後に 7 秒沈黙した場合、本題へ戻す一言を要求 |
| 再試行 | 人間に遮られた介入を最大 2 回、60 秒以内で再試行 |

### 3.3 現在の強み

- `trigger()` の呼び出しが `_run_agent_worker` に集約されている
- 応答生成の二重発火を防いでいる
- AI 発話中に人間が話した場合、AI が譲る
- `response.cancel` と `conversation.item.truncate` に対応している
- 「介入不要」の音声を再生前に破棄できる
- AI 声紋とテキスト類似度によるエコー除去がある
- クールダウン、ウォームアップ、相槌除外がある
- UI から議題と動作モードを変更できる

### 3.4 現在の問題

1. 発話数や沈黙が、そのまま介入検討の主要トリガーになっている
2. 問題の単発検出と継続検出を区別していない
3. 検出結果を共通形式で保持していない
4. 安全、理解、時間管理を専用に診断していない
5. 介入の強度や種類を明示的に選択していない
6. 介入後に問題が改善したかを評価していない
7. 会議目的に応じた中立性・内容介入方針がない
8. 音声 AI と議論グラフ型介入が別々に発火しうる
9. トリガー理由、モデル判断、実際の発言、効果が一つのログで追えない

---

## 4. 目標アーキテクチャ

```text
確定発話・沈黙・参加度
        ↓
   Signal Detectors
        ↓
FacilitationSignal の更新
        ↓
  Persistence / Severity
        ↓
 Intervention Coordinator
        ↓
    InterventionPlan
        ↓
   RealtimeAgent.trigger()
        ↓
      音声介入
        ↓
    Effect Evaluator
        ↓
 resolved / continue / escalate
```

重要な責務分離:

- **Detector**: 問題候補を検出する。自分では介入しない
- **Tracker**: 問題の継続性、重大度、再発、介入履歴を保持する
- **Policy**: 介入するか、何を優先するか、強度を決める
- **Renderer**: Realtime API に渡す指示文を組み立てる
- **Effect Evaluator**: 介入後の変化を評価する
- **RealtimeAgent**: 指示された内容を音声として安全に再生する

`RealtimeAgent` に診断ロジックを追加しない。Realtime API 接続、音声生成、
再生、割り込み、エコー対策に責務を限定する。

---

## 5. 観察・診断する軸

### 5.1 安全・尊重 `safety`

対象:

- 人身攻撃、侮辱、冷笑、威圧
- 同一人物への攻撃集中
- 発話を萎縮させる強い否定
- 合意済みルールへの明確な違反

方針:

- 高重大度なら継続確認や沈黙を待たずに介入可能
- 内容の正誤ではなく、発言の形式と場の安全だけを扱う
- 最初は人格と意見を分離する短いルール確認を行う

### 5.2 公平・参加機会 `participation`

対象:

- 発話時間、回数、文字数の偏り
- 長時間発言していない参加者
- 同一人物による連続発話
- 特定参加者への遮りの集中

現行の `participation_stats()` を再利用する。

### 5.3 理解 `understanding`

対象:

- 同じ用語を異なる意味で使っている
- 質問へ回答せず別論点へ移動している
- 相手の主張を誤って要約している
- 複数の論点が混ざっている
- 発言が拾われず宙に浮いている

最初の介入は、正解の提示ではなく明確化質問または言い換え確認とする。

### 5.4 焦点 `focus`

対象:

- 議題からの継続的な逸脱
- 一時的余談ではなく、別テーマへの移行
- 現在の論点と発話の対応が不明

現行の脱線検出を移行する。単発の `drift=true` で即介入するのではなく、
重大度が高い場合を除き、継続回数または継続時間を確認する。

### 5.5 進行・時間 `progress`

対象:

- 新しい情報を生まない主張の反復
- 長い沈黙
- 同じ対立の循環
- 決めるべき問いが不明
- 議論段階が切り替わらない

沈黙は介入理由そのものではなく、介入を発話する好機としても扱う。

### 5.6 収束・合意 `consensus`

対象:

- 合意点と相違点が整理されていない
- 合意表現はあるが、未確認の条件が残っている
- 決定後の次アクションが不明
- 表面的な同意と明示的な懸念が同時に存在する

初期実装では、直近発話からの合意・保留・反対シグナルのみを扱う。

### 5.7 明示的な支援要請 `help_request`

対象例:

- 「整理して」
- 「どう進めればいい」
- 「一度まとめて」
- ファシリテーターへの直接呼びかけ

明示的要請は通常の継続回数を満たさなくても介入候補にできる。

---

## 6. 共通データモデル

### 6.1 検出シグナル

```python
SignalKind = Literal[
    "safety",
    "participation",
    "understanding",
    "focus",
    "progress",
    "consensus",
    "help_request",
]


@dataclass(frozen=True)
class SignalEvidence:
    speaker: str | None
    text: str
    turn_id: int | None
    observed_at: float


@dataclass
class FacilitationSignal:
    id: str
    kind: SignalKind
    summary: str
    severity: float
    confidence: float
    first_seen_at: float
    last_seen_at: float
    consecutive_hits: int = 1
    involved_speakers: list[str] = field(default_factory=list)
    evidence: list[SignalEvidence] = field(default_factory=list)
```

`severity` は放置コスト、`confidence` は検出確度として分離する。

### 6.2 問題追跡状態

```python
@dataclass
class IssueState:
    signal: FacilitationSignal
    status: Literal["observing", "ready", "intervened", "resolved", "expired"]
    intervention_level: int = 0
    attempts: int = 0
    last_intervention_at: float | None = None
    effect_check_after_turn: int | None = None
    last_outcome: Literal["unknown", "improved", "unchanged", "worsened"] = "unknown"
```

### 6.3 介入計画

```python
InterventionType = Literal[
    "clarify",
    "summarize",
    "redirect",
    "invite",
    "process_check",
    "rule_reminder",
    "consensus_check",
    "next_action",
]


@dataclass(frozen=True)
class InterventionPlan:
    issue_id: str
    signal_kind: SignalKind
    intervention_type: InterventionType
    level: Literal[1, 2, 3]
    target_speaker: str | None
    instruction: str
    reason: str
    interrupt_allowed: bool = False
```

Realtime Agent には自由判断用の会話全体だけでなく、この `InterventionPlan` を
明示的に渡す。原則として一度の介入で一つの問題だけを扱う。

---

## 7. 介入レベル

### Level 1: 低侵襲

目的は参加者自身の自己修正を促すこと。

- 短い問い返し
- 言葉の意味の確認
- 要約の確認
- まだ出ていない視点を尋ねる
- 発言の少ない人への任意回答の声かけ

例:

```text
「いまの『効率』は、時間と費用のどちらを指していますか？」
```

### Level 2: 整理・プロセス介入

Level 1 後も問題が続いた場合に使用する。

- 論点を分ける
- 現在地を要約する
- 決める問いを再提示する
- 発話順や短いラウンドを提案する
- 合意点と相違点を整理する

例:

```text
「コストと安全性の二つが混ざっています。まずコストから整理しませんか。」
```

### Level 3: 強い介入

安全上の問題、または Level 2 でも改善しない場合に限定する。

- ルールの明示
- 一時停止の提案
- 個人ではなく意見を扱うよう要求
- 会議の進め方自体を確認する

例:

```text
「個人への評価ではなく、提案の内容について話しましょう。」
```

内容的な新知識や AI 自身の意見は、音声ファシリテーターの基本機能には含めない。

---

## 8. 介入ポリシー

この章は将来追加する診断軸を含む全体方針を示す。MVPでは `2.3`〜`2.8` の
決定事項を優先し、未実装軸による介入は行わない。

### 8.1 優先順位

```text
safety
  > help_request
  > understanding
  > focus
  > participation
  > progress
  > consensus
```

同じ優先度では、以下の順で比較する。

1. severity
2. confidence
3. consecutive_hits
4. 最終介入からの経過時間

### 8.2 発火条件の初期値

| シグナル | 全体方針 | MVP |
|---|---|---|
| safety | severity が高ければ 1 回で発火。通常は 2 回確認 | 未実装 |
| help_request | 明示的要請 1 回 | 採用 |
| understanding | 直近窓で 2 回以上、または質問未応答が継続 | 未実装 |
| focus | 原則2回連続検出 | 採用。高信頼でも1回発火はしない |
| participation | 数値ゲート + LLM 判定 + 沈黙の間 | 採用。判定自体も2回連続必要 |
| progress | 反復または停滞の継続を観測 | 観測のみ。自動発火しない |
| consensus | 最低発話数後、合意・保留シグナルが共存 | 未実装 |

値は設定オブジェクトへ集約し、コード内へ散在させない。

### 8.3 発話タイミング

- 将来は `safety` の高重大度だけ割り込みを許可しうる
- MVPでは診断軸にかかわらず人間へ割り込まない
- それ以外は原則として発話終了後の沈黙を待つ
- AI 応答中に人間が話し始めたら、現行どおり AI が譲る
- 一度の介入は原則 20 秒以内
- 介入後は最低クールダウンを置く
- 同一人物への連続介入を避ける

### 8.4 発話数トリガーの変更

`agent_trigger` は「介入する周期」ではなく「総合診断を実行する最大間隔」とする。

```text
現状:
N発話蓄積 → Realtimeモデルへ介入要否を問い合わせる

変更後:
N発話蓄積 → detector群を更新する
           → readyなIssueStateがある場合だけ介入計画を作る
```

---

## 9. 介入後の効果確認

介入後、既定で 2〜4 発話を観察して結果を評価する。

| 介入 | 改善の例 |
|---|---|
| redirect | 発話が議題または対象論点へ戻る |
| invite | 対象者が発話する、または明示的に辞退する |
| clarify | 用語の定義、言い換え、理解確認が行われる |
| summarize | 要約への同意・修正が返る |
| rule_reminder | 攻撃的表現が止まり、論点へ戻る |
| consensus_check | 合意、保留条件、反対点が明示される |

結果:

- `improved`: issue を resolved にする
- `unchanged`: 同じ level で 1 回まで再試行、または level を上げる
- `worsened`: 優先度を上げ、必要なら強い介入へ
- `unknown`: 観察を継続する

同じ介入文を繰り返さない。最大試行回数を超えた問題は、AIが会話を占有しないよう
`expired` にするか、人間ファシリテーター向け警告へ切り替える。

---

## 10. Realtime API 用プロンプト方針

ファシリテーターの基本制約:

- 中立を保ち、自分の賛否や結論を述べない
- 参加者の意見を正しい・間違いと評価しない
- 最初は質問、確認、要約から入る
- 一度に一つの問題だけ扱う
- `InterventionPlan` にない新しい論点を追加しない
- 対象者の回答を強制しない
- 安全上の問題以外では人間の発話を遮らない
- 問題が解消していれば「介入不要」を返す
- 前置きや自己説明をせず、20 秒以内で話す

将来的には自由文のシステムプロンプトだけに依存せず、介入計画を次のように渡す。

```text
[介入計画]
type: clarify
level: 1
target: 全員
reason: 「効率」の意味が参加者間で異なる可能性
instruction: 用語の意味を短い質問で確認する。答えを提示しない。
```

---

## 11. ファイル構造

既存の音声・STT・UIコードと、介入判断コードを分離する。

### Phase V0の実装構造

Phase V0では観測性だけを実装し、診断・調停・効果判定の抽象化は導入しない。

```text
src/das/asr/live/facilitation/
├── __init__.py
├── events.py
└── journal.py

tests/unit/live/facilitation/
├── __init__.py
├── test_events.py
└── test_journal.py
```

| ファイル | 責務 |
|---|---|
| `events.py` | 介入ID、イベント種別、発火理由、入力発話、モデル出力、再生有無の共通形式 |
| `journal.py` | 介入イベントを会議別のJSONLへ追記 |

介入イベントは次の順序で流す。

```text
_workers.py
  発火判断・intervention_id生成
      ↓
agents/_realtime.py
  Realtime API送信・再生・完了・中断・介入不要を通知
      ↓
_run_facilitator_event_worker
  受信スレッド外でJSONLへ保存
```

出力先:

```text
<meeting>.interventions.jsonl
```

既存の `<meeting>.diag.jsonl` は声紋診断用であり、介入イベントとは混在させない。

### Phase V1以降の予定構造

実際に該当Phaseを実装するときにファイルを追加する。未実装Detectorの空ファイルは
先に作らない。

```text
src/das/asr/live/facilitation/
├── __init__.py
├── events.py
├── journal.py
├── models.py
├── config.py
├── coordinator.py
├── issue_tracker.py
├── renderer.py
├── effect_evaluator.py
└── detectors/
    ├── __init__.py
    ├── focus.py
    ├── participation.py
    ├── help_request.py
    └── progress.py
```

| ファイル | 責務 |
|---|---|
| `models.py` | Signal、IssueState、InterventionPlan の型 |
| `config.py` | 閾値、観察窓、クールダウン、最大試行回数 |
| `coordinator.py` | シグナルの優先順位づけと介入計画決定 |
| `issue_tracker.py` | 問題の継続性、再発、介入履歴、状態遷移 |
| `renderer.py` | InterventionPlanからRealtime API用指示文を生成 |
| `effect_evaluator.py` | 介入後の効果判定 |
| `detectors/*` | 各診断軸の検出。介入は行わない |

既存ファイルの変更方針:

| ファイル | 変更内容 |
|---|---|
| `_workers.py` | 個別判定を減らし、Coordinatorの呼び出しと音声発火に限定 |
| `_session_state.py` | 介入ログ出力先とJournalを保持。将来はFacilitationRuntimeを保持 |
| `_bootstrap.py` | detector/coordinatorの組み立て、設定注入 |
| `_constants.py` | 音声・エコー定数のみ残し、介入閾値は `facilitation/config.py` へ |
| `agents/_realtime.py` | InterventionPlanを受け取る送信口を追加。診断は持たない |
| `_webapp.py` | 診断状態、直近介入理由、積極性設定の表示 |

---

## 12. テスト構造

Phase V0:

```text
tests/unit/live/facilitation/
├── test_events.py
└── test_journal.py
```

Phase V1以降は、実装したモジュールに対応するテストだけを追加する。

```text
tests/unit/live/facilitation/
├── test_models.py
├── test_issue_tracker.py
├── test_coordinator.py
├── test_renderer.py
├── test_effect_evaluator.py
└── detectors/
    ├── test_focus.py
    ├── test_participation.py
    ├── test_help_request.py
    └── test_progress.py
```

既存の以下のテストは維持する。

- `test_realtime_agent.py`: Realtime API、介入不要、割り込み、再試行
- `test_agent_worker.py`: ターンテイキングと最終発火
- `test_modes.py`: 動作モード切り替え
- `test_ui_api.py`: UI API
- `test_participation.py`: 発話量計算

テスト原則:

- Detector は純粋関数または状態を明示した小さなクラスにする
- LLM 判定は構造化出力をモックする
- Coordinator はLLMなしで決定的にテストできるようにする
- 時刻は `time.monotonic()` を直接呼ばず Clock を注入可能にする
- 各段階で `tests/unit/live` を全件成功させる
- 実機テストは自動テストと分け、チェックリスト化する

---

## 13. 実装フェーズ

### Phase V0: 観測性の確保

目的: 現行動作を変えず、比較可能なログを残す。

- [x] 介入イベントの共通ログ形式を定義
- [x] trigger理由、入力発話、モデル出力、実際の再生有無を記録
- [x] 介入開始・終了・中断・介入不要を同じ `intervention_id` で追跡
- [x] JSONLで確認可能にする

V0イベント種別:

- `trigger_requested`
- `trigger_suppressed`
- `response_requested`
- `speech_started`
- `speech_completed`
- `utterance_completed`
- `response_completed`
- `interrupted`
- `noop`
- `error`

V0発火理由:

- `count`
- `silence`
- `drift`
- `invite`
- `stall`
- `retry`

`signal_kind`、`severity`、`confidence`、`issue_id`、介入効果はV0では
未計算であり、V1/V2で追加する。

完了条件:

- 1回の介入について「なぜ発火し、何を話し、どう終わったか」を追跡できる

### Phase V1: 共通モデルと調停器

目的: 既存トリガーを新しい共通形式へ移す。

- [ ] `FacilitationSignal`、`IssueState`、`InterventionPlan` を実装
- [ ] TrackerとCoordinatorを実装
- [ ] 既存の脱線・参加偏りをDetector化
- [ ] `_run_agent_worker` はCoordinatorの結果だけを発火
- [ ] 発話数を診断周期へ変更

完了条件:

- 脱線と参加偏りが単一Coordinatorで優先順位づけされる
- 単発検出と継続検出を区別できる

### Phase V2: 効果確認

目的: 介入後の「再観察」を実装する。

- [ ] redirect後の焦点復帰判定
- [ ] invite後の対象者発話判定
- [ ] improved / unchanged / worsened の状態遷移
- [ ] 同じ問題への再試行と最大試行回数
- [ ] 介入レベルの段階的上昇

完了条件:

- 脱線介入と声かけについて、効果を自動判定できる

### Phase V3: 安全・尊重

目的: 放置コストが高い問題に対応する。

- [ ] safety detector
- [ ] severityによる即時介入判定
- [ ] 中立的なrule reminder
- [ ] 高重大度だけ割り込みを許可
- [ ] 誤検知時の抑制と監査ログ

完了条件:

- 安全問題と通常問題で発話タイミングを分けられる

### Phase V4: 理解・進行

目的: 内容へ踏み込みすぎず議論の質を支える。

- [ ] 用語ずれ・論点混同 detector
- [ ] 質問未応答 detector
- [ ] 反復・停滞 detector
- [ ] clarify / summarize / process_check の計画生成

完了条件:

- 最初は明確化、継続時は整理という段階制御が動く

### Phase V5: 収束・合意

目的: 議論後半の整理と次アクション確認に対応する。

- [ ] 合意・保留・反対シグナルの検出
- [ ] 合意点と相違点の確認
- [ ] next action確認
- [ ] 見せかけの合意への注意喚起

完了条件:

- 合意をAIが作るのではなく、参加者に確認できる

### Phase V6: UI・設定・実機評価

- [ ] UIに直近シグナル、介入理由、level、効果判定を表示
- [ ] 会議種別と中立性設定を追加
- [ ] detector単位のON/OFF
- [ ] controlled / standard / active を新しい設定へ移行
- [ ] 2〜3人の実機シナリオテスト
- [ ] 過介入、誤介入、割り込み、レイテンシを記録

完了条件:

- 設定変更と介入理由を利用者が確認できる
- 代表シナリオで再現可能な実機評価結果が残る

---

## 14. 軸別動作確認オプション

診断軸・発火条件ごとの挙動を単独で確認できるよう、実装時にCLIオプションを追加する。
この節はオプション仕様案であり、現時点では未実装。

### 14.1 有効にする診断軸

```bash
--facilitation-axes focus,participation,help_request
```

指定可能な値:

- `focus`
- `participation`
- `help_request`
- `progress`
- `all`

既定値:

```text
focus,participation,help_request,progress
```

ただし `progress` はMVPでは観測のみで、自動介入しない。

単独確認例:

```bash
# 脱線だけを診断・介入
uv run python -m das.asr.live \
  --agent \
  --topic "来期の開発計画" \
  --facilitation-axes focus

# 発話量の偏りだけを診断・介入
uv run python -m das.asr.live \
  --agent \
  --facilitation-axes participation

# 停滞を診断するが、発話させずログだけ確認
uv run python -m das.asr.live \
  --agent \
  --facilitation-axes progress \
  --facilitation-observe-only
```

### 14.2 観測専用モード

```bash
--facilitation-observe-only
```

このモードではDetector、Tracker、Coordinatorまで動作させるが、音声介入は行わない。
検出結果、発火条件成立、生成予定だった介入計画をログへ残す。

新しい診断軸は、最初にこのモードで誤検知率を確認してから音声介入を有効にする。

### 14.3 閾値上書き

実験・チューニング用途として、MVPでは次の上書きを許可する。

```bash
--focus-consecutive-hits 2
--participation-consecutive-hits 2
--effect-window-turns 3
--facilitation-pause-seconds 2.0
--facilitation-cooldown-seconds 20.0
--facilitation-max-per-minute 2
```

通常利用者向けにはプリセットを優先し、細かい閾値は詳細オプションとして扱う。

```bash
--facilitation-profile controlled
--facilitation-profile standard
--facilitation-profile active
```

プリセットと個別上書きを同時指定した場合は、個別上書きを優先する。

### 14.4 テスト用の発火強制

実音声・E2E確認用に、診断を偽装せず「指定した介入計画だけを一度試す」開発者向け
オプションを将来用意する。

```bash
--facilitation-demo redirect
--facilitation-demo invite
--facilitation-demo summarize
```

これはDetectorの精度評価には使用しない。音声、語調、割り込み、UI表示だけを
確認するための機能とする。本番モードでは無効化する。

---

## 15. 診断軸・発火条件・問題・効果の対応表

### 15.1 MVP

| 診断軸 | 想定する問題 | 発火条件 | 最初の介入 | 期待する効果 | 効果判定 | 過介入リスク |
|---|---|---|---|---|---|---|
| `focus` | 議題と無関係な話題が継続し、本題へ戻らない | 脱線を2回連続検出 | Level 1の短い問い。「いまの話は議題のどの部分に関係しますか？」 | 参加者自身が関連を説明するか、本題へ戻る | 介入後3発話で議題との関連が回復 | 有用な余談や具体例を脱線扱いする |
| `participation` | 一部参加者だけが話し、他の参加者の意見が出ない | 8発話以降、偏りを2回連続判定し、2秒の沈黙がある | 対象者へ任意回答のinvite | 発話機会が増え、参加の偏りが緩和する | 対象者が発話または辞退。発話シェアの変化は補助指標 | 発言したくない人へ圧力を与える |
| `help_request` | 参加者が整理や進行方法に困っている | 明示的な支援要請1回 | 要請に対応する短い要約または進め方確認 | 参加者が次に扱う論点を選べる | 要約への同意・修正、または議論再開 | 依頼範囲を超えて議論を誘導する |
| `progress` | 長い沈黙、主張の反復、進展不足 | MVPでは自動発火しない | なし。観測ログのみ | 検出精度と閾値を検証する | 人手で停滞の有無と検出結果を比較 | 通常の思考時間や慎重な反復を問題扱いする |

### 15.2 将来追加する軸

| 診断軸 | 想定する問題 | 初期介入 | MVPでの扱い |
|---|---|---|---|
| `safety` | 人身攻撃、侮辱、威圧、冷笑 | 中立的なルール確認 | 未実装 |
| `understanding` | 用語ずれ、論点混同、質問未応答 | 明確化質問 | 未実装 |
| `consensus` | 表面的合意、条件・懸念の未確認 | 合意点・保留点の確認 | 未実装 |

### 15.3 介入レベルと問題別の変化

| 軸 | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| `focus` | 議題との関連を質問 | 現在地を要約し、扱う問いを再提示 | 一度区切り、進め方自体を確認する候補として記録。MVPでは発話しない |
| `participation` | 任意回答で意見を求める | 個人への再指名を避け、全員への問いへ切り替える | 自動介入を停止し、未改善として記録 |
| `help_request` | 求められた範囲を短く支援 | さらに整理が必要かを確認する | 自動介入を停止し、未改善として記録 |
| `progress` | MVPでは介入しない | MVPでは介入しない | MVPでは介入しない |

---

## 16. Phase V0 介入ログ仕様

MVPでは以下をJSONLに記録する。発話全文を重複保存せず、原則としてturn IDと
必要最小限の抜粋を記録する。

```json
{
  "intervention_id": "int_...",
  "detected_at": "2026-06-25T12:00:00Z",
  "signal_kind": "focus",
  "severity": 0.6,
  "confidence": 0.9,
  "consecutive_hits": 2,
  "evidence_turn_ids": [8, 9],
  "evidence_excerpt": ["...", "..."],
  "issue_id": "issue_...",
  "intervention_level": 1,
  "intervention_type": "redirect",
  "target_speaker": null,
  "trigger_reason": "脱線を2回連続検出",
  "planned_instruction": "議題との関連を短い質問で確認する",
  "model_output": "いまの話は、今日の議題のどの部分に関係しますか？",
  "decision": "speak",
  "playback_started_at": "...",
  "playback_ended_at": "...",
  "interrupted": false,
  "effect_check_after_turn": 12,
  "effect_outcome": "improved"
}
```

`decision`:

- `speak`
- `noop`
- `suppressed_cooldown`
- `suppressed_rate_limit`
- `observe_only`
- `expired`

ログに残す最低項目:

- 発火した診断軸
- severity / confidence / 継続回数
- 根拠ターン
- 発火・抑制理由
- 介入レベルと種類
- モデルが生成した発言
- 再生開始・終了
- 人間による中断
- 効果判定

---

## 17. 実機評価シナリオ

### 17.1 MVPで実施するシナリオ

| シナリオ | 期待動作 |
|---|---|
| 健全な議論 | 原則として黙る |
| 一時的な余談 | 介入しない |
| 継続的な脱線 | 低侵襲なredirect |
| 1人が発話を支配 | 沈黙の間に別参加者を任意回答でinvite |
| 「一度整理して」と依頼 | 依頼範囲内の短い要約 |
| 長い沈黙や主張の反復 | 音声介入せず、progressログだけを残す |
| AI介入へ人が被せる | AIが譲り、必要性を再評価 |
| AI音声をマイクが拾う | 再トリガーしない |

### 17.2 将来軸で追加するシナリオ

| シナリオ | 期待動作 |
|---|---|
| 用語の意味がずれる | 短いclarify |
| 同じ主張を反復 | 要約または決める問いを再提示 |
| 人身攻撃 | 速やかにrule reminder |
| 表面的な合意 | 条件・懸念の確認 |

記録する指標:

- 検出から介入開始までの時間
- 発火したシグナルと証拠
- 介入不要率
- 参加者による割り込み率
- 同一問題への再介入回数
- 効果判定結果
- 人手評価による適切性、タイミング、中立性、しつこさ

---

## 18. 設計上の制約

### 中立性

音声ファシリテーターは、参加者の代わりに結論を決めない。自分の意見、賛否、
価値判断を述べない。内容知識を加える場合は別機能として明示する。

### プライバシー

診断ログに発話全文を残す場合は、保存設定と保持期間を明示する。将来的には
証拠テキストを短くする、またはハッシュ・turn ID参照にする。

### 誤検知

安全以外は単発検出で強く介入しない。LLMの判定だけでなく、継続性、数値指標、
クールダウンを組み合わせる。

### レイテンシ

Detectorを逐次直列実行しない。決定的Detectorを先に実行し、LLMが必要なものだけ
まとめて判定する。介入判断が会話の文脈に間に合わない場合は破棄する。

### 障害時

診断やRealtime APIが失敗しても、文字起こしと議事録保存を継続する。
ファシリテーションは fail-open、つまり黙る側へ倒す。

---

## 19. 未決事項

- [ ] 安全判定を単一LLMに任せるか、ルール＋LLMの二段階にするか
- [ ] `progress` の自動発火条件
- [ ] 介入効果を決定的に判定する範囲とLLMを使う範囲
- [ ] 会議種別の分類: 公共熟議、業務会議、教育、自由対話
- [x] MVPでは内容介入を禁止する
- [ ] 高重大度safetyで人間発話へ割り込む条件
- [ ] 診断周期とコスト上限
- [ ] Realtimeモデルに渡す文脈窓
- [ ] UIで参加者に介入理由をどこまで開示するか
- [ ] 音声AIとグラフ型テキスト介入を同時利用する場合の排他制御

---

## 20. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-06-25 | 初版。現状、目標構造、診断軸、実装フェーズを定義 |
| 2026-06-25 | MVPを中立的な司会役、人間2〜3名、focus/participation/help_request/progressに限定。発火条件、介入強度、効果判定、過介入上限、CLI案、対応表、ログ仕様を決定 |
