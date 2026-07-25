# 引き継ぎ: 二重帳簿の根治（人物鋳造の一本化） — 2026-07-25

前セッション（Claude, 2026-07-22〜25）からの引き継ぎ。**次セッションの任務は
「話者IDの二重帳簿の根治」**。ユーザー承認済み。着手前にこの文書全体と
handoff_2026-07-14_unregistered_speakers.md の §17〜§19 を読むこと。

## 1. 問題（実測で確定済み）

ハイブリッド構成では、同じ人間が2つの帳簿に別名義で載る:

- **クラスタ帳簿**: diarizationクラスタ → `@diar:N`（`diarization_speaker_keys`）
- **声紋帳簿**: STTラベルの声紋蓄積が鋳造する `人物N`（`VoiceProfiles._commit_profile`）

実会話ログでの実証（声紋を再計算して突き合わせた結果）:

| セッション | 事実 |
|---|---|
| 2026-07-25_1723 (2人) | 参加者A(@diar:1) と 参加者B(人物1) が **類似0.613=同一人物**。1人が2席を独占 → 2人目は席なしで大半未確定、終盤に人物2（=2人目、別人）が「参加者C」として出現 |
| 2026-07-25_1641 (2人) | 人物1 と @diar:2 が **類似0.702=同一人物**。同じ構図 |

- 別人ペアの類似は 0.03〜0.31、同一人物ペアは 0.61〜0.70 と**綺麗に分離**。
- 既存の統合機構（クラスタ確定, `ClusterVoiceNamer.observe` → 確定バー
  `PYANNOTE_CLUSTER_CONFIRM_MIN_SIM=0.65`）は 1723 で類似0.613を**0.04差で
  見送り続けた**。バー0.65は過去の誤確定事故（sim0.54→誤帰属37件, handoff
  §15.9）由来なので単純に下げない。
- 被害: 「幻の3人目」ではなく**席の食い潰し→締め出された実在者の発話が
  未確定化**。現行の未確定の主因の一つ。統一席ルール（bc56b8b）で
  「参加者Cの出現」は止めたが、席の二重取りそのものは残っている。

## 2. 承認済みの方針

**根治 = 人物の鋳造窓口をクラスタ経由に一本化する**。設計候補は2つあり、
Phase 0 の数字で決める:

- **案A（鋳造一本化）**: ハイブリッド時、STTラベル側の鋳造
  （`_commit_profile` の新規人物作成）を停止し（照合・分類は継続）、
  クラスタ側に鋳造を移す（クラスタに十分なクリーン音声が溜まり、既存の
  誰とも dedupe 以上で一致しなければ、クラスタ代表声紋で人物Nを鋳造して
  即 confirm/rekey）。1人=1クラスタ=1戸籍=1席。
- **案B（鋳造時リンク）**: ラベル側の鋳造は残すが、鋳造の瞬間に
  「新人物のプロファイル vs 席持ちクラスタの蓄積声紋」を**対称比較**し、
  閾値（実測分離から0.45前後が候補）以上なら新席を作らずそのクラスタ
  IDに統合する。変更が局所的で、確定バー0.65の一方向照合より情報量の
  多い比較になる。

## 3. 次セッションの手順（Phase 0 → 実装）

1. **Phase 0（反実仮想測定, まずこれ）**: GT付き記録で案A/案Bを
   オフライン採点し、実質正解率・誤帰属・未確定の変化を出す。材料:
   - GT14本（replay系: eval/gt_*.json。ただし replay は声紋層のみ再現、
     クラスタ層は再現不可＝diag記録からの系列再生で補う。前セッションで
     使った手法: diag の key/final_key 系列を SessionState 実物に流す
     再現・反実仮想。tests/unit/live/test_session_state.py の
     counterfactual テストが実装例）
   - 実会話5本（transcripts/ の 2026-07-22_223337, 2026-07-25_1534/1545/
     1641/1723。GT無しだが、声紋突き合わせで同一人物判定が可能なことは
     実証済み）
2. 数字をユーザーに提示して案を選び、**承認を得てから**実装。
3. 実装は opt-in から（既定挙動不変で入れ、ライブ検証後に既定化）。
   単独モード（Soniox単独/pyannote単独）は挙動不変を厳守。
4. ゲート: pytest 全green（既知flake `test_agent_worker.py::
   test_structuring_checker_rejudges_after_pending_reset` は除外可）。
   replay基準（§18.10）は案Bなら不変のはず。案Aは声紋層の鋳造が変わる
   ため**基準の再構築が必要**（変更前に必ずユーザーに明示して承認）。

## 4. 作業ルール（このプロジェクトの掟。全て実運用で確立済み）

- **実験・検証・挙動変更は着手前にユーザー承認**（過去に無断実験で叱責あり）
- 1変更=1コミット。コミットメッセージに経緯と数字を書く
- 誤帰属＞未確定の優先。不可逆操作は高確信のみ。データなき閾値変更はしない
- サブエージェントを使う場合は安いモデル（haiku）。ただしユーザーは
  「自分で調査してほしい」と言うことがある——その指示があれば自分で読む
- ruff: 既存エラー（_bootstrap/_session_state の I001、_workers の
  UP037/RUF046、test_diarization の N806×4）以外を増やさない

## 5. 環境とファイルの場所

- **Mac**: `~/discussion-support`（デバイスVM経由: `/sessions/<id>/mnt/discussion-support`）。
  VMは**ファイル削除不可**（rm不可→`_to_delete/`へmv）。**git merge等の
  index操作も不可**（ロック残骸は `_to_delete/git_tmp/` へ）。
  受け渡しは「クラウドで bundle 作成 → SendUserFile → device_commit_files
  → device_bash で `git fetch <bundle> +main:refs/heads/main-updated`。
  merge はユーザーがMacで実行」
- **クラウド作業環境（セッションごとに使い捨て）**: リポジトリは毎回
  Mac から bundle で持ち込む: device_bash で
  `git bundle create /Users/matsuirei/discussion-support/repo.bundle main`
  → stage → クラウドで clone。`.venv` は `uv sync` 等で再構築
  （torch hub の ReDimNet 重みは HF ミラー。過去の手順:
  handoff_2026-07-14 §環境 / このセッションでは /root/.cache に構築済み
  だが次セッションには残らない）
- **最新コミット（この文書を含む時点の main）**: 監査A〜E修正済み。
  ユーザーのMacで `git merge main-updated` が未実行の可能性あり——
  最初に `git log --oneline -3 main` を確認すること
- 実会話ログの分析済み中間データ（声紋類似の数字）は本文書 §1 に記載。
  /tmp の切り出しwav等は消えるため、必要なら再抽出（デバイス側で
  python3 により diag の ms/end から wav を切り出す手法を使用）

## 6. 関連ファイル

- 帰属フロー: `src/das/asr/live/_attribution.py`（モジュールdocstringに全体図）
- 声紋・鋳造: `src/das/asr/live/_voice_profiles.py`（`_commit_profile` が鋳造）
- クラスタ命名: `src/das/asr/live/_cluster_naming.py`（observe/確定バー0.65）
- 席・帳簿: `src/das/asr/live/_session_state.py`（constrain=統一席ルール、
  rekey=統合の単一入口、key_for_diarization_speaker=ヒステリシス）
- 採点: `eval/replay_attribution.py`, `eval/_gtlib.py`, `eval/eval_speaker_gt.py`
- 経緯の全記録: `docs/design/handoff_2026-07-14_unregistered_speakers.md`
  （§17統合, §18リファクタ〜sweep, §19介入修正）,
  `docs/design/sortformer_feasibility_2026-07-22.md`（土台交換の検証と結論）

## 7. 未完了・保留中の関連事項

- 実会話マイク直の本命検証（runbook: docs/design/mic_verification_runbook_2026-07.md）
  は録音待ちのまま。根治後の再測定と兼ねられる
- Sortformer opt-in（--diarization sortformer）はMacで動作確認済み。
  クリーン音源で強く、マイク残響で弱い（feasibility文書参照）
- 介入は2レーンとも同一内容の再発火抑止済み（Realtimeレーン=
  duplicate_content 10分窓、グラフレーン=価値ゲート+本文10分窓）
