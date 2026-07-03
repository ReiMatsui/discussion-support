# 設計書: 統合AF×ライブ介入の接続（H1）＋ロジック改良の織り込み

実施者向け。背景: `docs/research/logic_review_2026-07.md`（改良トップ5）、`docs/review-2026-07-02/01_live_pipeline.md` 再設計案A、`08_deep_review_summary.md`。品質基準は従来どおり（コミット分割・全テスト通過・ruffクリーン・mypy増分ゼロ・対症療法禁止）。

## 0. ゴールと非ゴール

**ゴール**: 対面ライブセッション（`das.asr.live`）で、統合議論グラフ（AF）に基づく介入（L1個別・L2俯瞰）が既存の FacilitationController 採否・タイミング制御・RealtimeAgent 発話を通って動く。研究の貢献②③が対面モダリティで成立する。

**非ゴール（今回やらない）**: fact/drift チェッカーのAFへの吸収（統合後の様子を見て別計画）、画面（/board）へのL1配信（UI刷新と一体で後日）、介入効果に基づく適応制御（計測のみ実施）。

**設計原則**: what/whom は AF 状態のみから決める（`decide_intervention`）。when は会話の物理（Controller の pause/cooldown/floor）。how は RealtimeAgent のプロンプト。この三層分離を崩さない。

## 1. アーキテクチャ

```
確定発話(records) ──┐
                    ├─ 既存: triage/fact/drift/participation checkers
                    └─ 新設: _af_runtime (Orchestrator+GraphStore 常駐)
                         │  extraction → linking （非同期・逐次）
                         ▼
                    AF checker（数秒周期）
                         │  decide_intervention(history, store)
                         │  → InterventionCandidate(kind="af_l1"/"af_l2")
                         ▼
              既存 FacilitationController（採否・タイミング一元裁定）
                         ▼
              既存 RealtimeAgent（音声化・barge-in・エコー管理）
```

新設モジュール: `src/das/asr/live/_af_runtime.py`
- ライブセッション開始時に `NetworkXGraphStore`＋`Orchestrator` を生成し保持
- 専用スレッドが records カーソル（meeting epoch ガード必須・既存パターン踏襲）で確定発話を取り込み、extraction→linking を実行。話者名は `intervention_speaker_name`（未確定は汎用名）
- 事前文書がある場合（`--docs` 相当の既存経路）は開始時に evidence 投入
- スナップショットを終了時と定期（60s デバウンス）に transcripts/ へ保存（既存の可視化・run-session 資産と互換の形式）

## 2. ロジック改良の織り込み（logic_review トップ5）

### 2.1 アクティブ窓（改良1, A5/B3/C13）
- `das/graph/schema.py` の Node に `turn_index: int`（会議内の確定発話連番）を追加（evidence は投入時の turn_index）。後方互換: default 0
- `FacilitationAgent.decide_intervention` に `active_window: int = 12` を導入し、**L1候補・偏り検知・stalled検知はアクティブ窓内のノード/エッジに限定**して計算する（窓外は L2 の俯瞰でのみ参照）
- 偏り検知は累積比率を廃し「窓内 support/attack 件数」の条件式に置換。priority の乗算係数（1.3/0.7/1.2/0.85）は撤廃し、ソートは (種別優先度, confidence) のみ
- テスト: 窓外の古い未応答攻撃が L1 候補にならない／窓内なら候補になる

### 2.2 L1価値ゲート（改良2, B1）
L1候補は次を**すべて**満たすときだけ出す（決定的判定、LLM追加なし）:
1. 緊張: 対象が「未応答の攻撃」または「根拠なし主張への evidence」である（単なる類似・支持の並記は出さない）
2. 新規性: 提示済みでない（`_af_runtime` が提示済みノードID集合を保持）かつ、source_text の主要部が直近 transcript に既出でない（既存 citation 照合ロジックの部分一致を流用）
3. 鮮度: 対象主張がアクティブ窓内
価値ゲートで落とした候補は理由コード付きでログ（`af_l1_skip: no_tension|already_presented|stale`）。**「なぜ出さなかったか」は一次データ**。

### 2.3 重複claimのsoft-merge（改良3, A2）
- `das/graph` に `cluster_id` を導入: linking が既に計算する embedding を再利用し、類似度閾値（例 0.9）以上の既存 claim と同クラスタに割当（非破壊・ノードは残す）
- 偏り・weak_claims・応答率の計算をクラスタ単位に変更。L1 新規性判定にも使用（同クラスタ提示済みなら出さない）
- 独立コミット。live 統合前に eval 側テストで検証可能

### 2.4 生成先行・再生ゲート（改良4, C-L2）
- `RealtimeAgent.trigger(..., hold_playback=True)` を追加: 応答生成は即開始するが、音声再生キューの解放を「フロア条件成立」まで保留。worker は沈黙 0.3s 時点で生成開始し、pause 成立で `agent.release_playback()`、成立前に新規発話が来たら `agent.cancel_held()`（response.cancel、リトライ扱いにしない）
- 対象はまず af_l1/af_l2/summarize（遅延に敏感でない種別から）。fact/manual は従来どおり（即時性優先で挙動を変えない）
- 実測ログ: `hold_to_release_ms` を timing metadata に追加
- 独立コミットで、AF統合と並行可能

### 2.5 介入ノードと応答エッジ（改良5, B4）— 計測のみ
- 配信された af 介入を `intervention` ノードとして store に追加し、提示したノードへ `presents` エッジを張る
- 以降の linking 実行時、新規発話ノードと intervention ノードの類似が閾値超なら `responds_to` エッジ（=受容の痕跡）。制御には使わず、snapshot と interventions.jsonl に記録するだけ
- これがライブ版 citation_rate になる（段階Cの受容性指標）

## 3. Controller への組み込み

- `Kind` に `"af_l1"`, `"af_l2"` を追加。`_KIND_POLICY`:
  - `af_l1`: priority 4（retry の後・summarize の前）、pause 1.5s、kind cooldown 20s、wait_for_pause
  - `af_l2`: priority 6、pause 2.0s、cooldown 60s、**global scope**（俯瞰は頻発させない）
- AF checker は候補を `_PendingInterventions` に積む（fact 同様 TTL 必須: af_l1 は 45s——アクティブ窓と整合、af_l2 は 90s）
- summarize と af_l2 の関係: **af_l2 が生成可能（グラフが十分育っている）なら summarize 候補より優先**（同 tick に両方あれば priority で af_l2 が勝つ設定にし、summarize は「AFがまだ薄い序盤の代替」と位置づける）
- dispatch: 採択時 `agent.trigger(af_presentation=payload)`。payload は関係ラベル付き提示文（`InterventionDecision` の items を整文したもの。整文は既存 L2 経路の `facilitation` 側ヘルパーを流用し、**呼び出し側複製を作らない**——レビュー02の既知問題を悪化させない）

## 4. 実装フェーズ（コミット順）

| # | 内容 | 依存 |
|---|---|---|
| 1 | schema: turn_index＋cluster_id（graph層、eval側テストで検証） | なし |
| 2 | facilitation: アクティブ窓＋窓内偏り検知＋係数撤廃（eval側で回帰確認） | 1 |
| 3 | `_af_runtime.py`: store/orchestrator 常駐＋発話取り込み＋snapshot保存（**介入なし**、レイテンシ計測ログ付き） | 1 |
| 4 | AF checker＋価値ゲート＋Controller統合（af_l1/af_l2 発話まで通す） | 2,3 |
| 5 | 介入ノード・応答エッジの計測 | 4 |
| 6 | 生成先行・再生ゲート（並行可） | なし |

フェーズ3完了時に必ず実測すること: 発話確定→ノード追加→エッジ追加の遅延分布（p50/p90）。**エッジ追加が10秒を超えるようなら、フェーズ4の前にアクティブ窓ベースの linking 対象絞り込み（新ノード×窓内ノードのみ照合）で最適化する**。

## 5. テスト戦略

- `_af_runtime`: フェイク extraction/linking（既存 eval テストの AsyncMock 資産流用）でカーソル・epoch・snapshot を検証
- AF checker: フェイク store 上で「窓内未応答攻撃→af_l1 候補」「提示済み→skip」「窓外→skip」
- Controller: af_l1/af_l2 の policy・優先関係（vs summarize/fact）のテストを test_facilitation_controller に追加
- 再生ゲート: hold→release/cancel の状態遷移（フェイクWS基盤流用）
- 統合スモーク: `--simulate` シナリオで af_l1 が発火することを手動確認（実施者は API キーなしでも単体が全部通ることを保証）

## 6. 研究文書への反映（実装完了時）

- RESEARCH.md の貢献③を「what/whom は AF、when は会話物理」の二層記述に更新
- research_plan のトラックB「AF×ライブ統合」を本設計書参照に差し替え、RQ対応表を af 介入前提で更新
