# 引き継ぎ: 登録者ゼロでも破綻しない話者帰属（2026-07-14）

> 作成: Claude Fable 5 (claude-fable-5), 2026-07-14。次セッションへの作業指示書。

## 1. 現状（ブランチ try/pyannote-live1、テスト63件パス）

ハイブリッド話者帰属を実装済み: Soniox=文字起こし／pyannote Live-1=クラスタリング／
声紋照合=クラスタ単位の名前付け（`_cluster_naming.py` の ClusterVoiceNamer、
`--diarization pyannote --vp-cluster-naming` で有効化）。
根拠データ: 盲検裁定2セッションで pyannoteクラスタ一貫性78% vs 現行の断片単位名前付け44%
（詳細は docs/design/pyannote_live1_trial_2026-07-09.md §8-9）。

## 2. 問題（今回の実地テストで判明）

3人の会話（YouTube動画をスピーカー再生、**声紋登録者ゼロ**、n_prof=1は本人のみ）で:
- 参加者A〜Gの7ラベルに膨張、同一人物が別人物として分裂
- 診断（transcripts/2026-07-14_1132.diag.jsonl）: ハイブリッドは動作していたが、
  (a) 登録プロファイルがないため名前付け層が無力、
  (b) 未照合クラスタがヒステリシス（累積3秒）を通過するたび「参加者X」に自動昇格、
  (c) スピーカー再生の圧縮・残響音声でクラスタ自体も割れやすい

ユーザー要件: **登録者ゼロは頻出ケースなので、この条件でちゃんと動くこと**。本質的かつシンプルに。

## 3. 設計方針（この順で実装）

登録者ゼロで提供すべき価値は「名前」ではなく**ラベルの一貫性**（同じ人が同じ参加者Xであり続ける）。
鍵は、声紋エンジンを「登録名との照合」だけでなく**クラスタ間の名寄せ**に使うこと。

1. **クラスタ間名寄せ（本質）**: 新クラスタの蓄積埋め込みを、既存の全クラスタ（名前付き＋未名の参加者X）
   の埋め込みと照合し、類似度が閾値以上なら**新規参加者を作らず既存に統合**する。
   これで (a) pyannoteのクラスタ分裂、(b) 再接続後のラベル空間変化、(c) 序盤の揺れ、が全て
   同じ機構で吸収される。ClusterVoiceNamer に蓄積埋め込みが既にあるので追加コストは小さい
2. **昇格の厳格化**: 未名クラスタの「参加者X」昇格は、名寄せで既存に統合できなかった場合のみ。
   `--diarization-max-speakers N` 指定時は参加者総数がNを超えないよう、超過分は最も近い既存参加者へ
   統合（それも不可なら未確定のまま）
3. **議事録の遡及統合**: クラスタ統合が起きたら、吸収された側の過去レコードを既存の rekey/リネーム
   機構で統合先ラベルに付け替える
4. 閾値は既存の声紋照合閾値（vp_match系）を流用し、新規定数は最小限に

やらないこと: 既存モード（Soniox単独、pyannote単独）の挙動変更、UI刷新、
オープンセット登録の高度化（masterplan §3 のスコープ管理を守る）。

## 4. 検証プロトコル（実装後にユーザーが実施）

1. `uv run pytest -q`（全体、既存63件＋新規）
2. YouTube 3人会話動画をスピーカー再生で5分（登録者ゼロ）:
   **合格基準 = 参加者ラベル総数が5以下、かつ主要話者の発話が概ね同一ラベルに集まる**
3. 本人＋登録済み1名の実会話: 登録名2つ＋未確定のみになること
4. 単独発話: 参加者が増えないこと（前回確認済みのヒステリシスの回帰確認）

## 5. 読むべきファイル

- src/das/asr/live/_cluster_naming.py（ClusterVoiceNamer、蓄積・照合の本体）
- src/das/asr/live/_session_state.py（key_for_diarization_speaker のヒステリシス、rekey）
- src/das/asr/live/_recv_loop.py（flush での配線）、_voice_profiles.py（match_profile）
- docs/design/pyannote_live1_trial_2026-07-09.md §8-9（経緯と設計根拠）
- 診断ログの読み方: transcripts/*.diag.jsonl の kind（声紋一致/蓄積中）と key（@diar:N）

## 6. 作業ルール

- ブランチ try/pyannote-live1 で作業、1機能=1コミット、テスト必須
- 挙動を変える箇所には設計根拠コメント（本ドキュメント参照の形式）
- 完了したら本ドキュメントに結果を追記し、検証プロトコルのコマンドをユーザーに提示

## 7. 実装結果（2026-07-14、Claude Fable 5 追記）

§3 を実装完了。コミット:

- `6dc5a65` feat(live): クラスタ間名寄せ（未照合クラスタを既存クラスタの声紋埋め込みへ統合）
- `e13e50e` feat(live): flush配線にクラスタ名寄せを反映（遡及統合＋max-speakers超過時の最近傍統合）

実装の要点:

- **名寄せ（§3-1）**: ClusterVoiceNamer に `_embeddings`（canonicalクラスタ→代表埋め込み）と
  `_aliases`（吸収→canonical）を追加。`observe()` で match_profile 不成立時に
  `tracker.embed()`（新設の公開API、`_embed` へ委譲）で埋め込みを取り、他クラスタと
  コサイン比較、`tracker.dedupe` 閾値以上で統合。新規定数ゼロ（§3-4）。
- **昇格の厳格化（§3-2）**: `_recv_loop.py` の匿名キー発行を `_merged_diarization_speaker_key()`
  経由に変更（cluster_namer 有効時のみ）。名寄せ成立クラスタは canonical のキーへ帰属。
  `--diarization-max-speakers` 到達時は `nearest_cluster()`（閾値なし最近傍）の既存キーへ統合、
  不可なら従来どおり constrain で未確定。判定用に `SessionState.human_slot_budget_exhausted()` を新設。
- **遡及統合（§3-3）**: 吸収側に発行済みの @diar:N があれば既存 `rekey()` で過去レコードごと
  canonical キーへ付け替え。声紋確定名への昇格時も raw/canonical 両キーを rekey し
  `diarization_speaker_keys` を確定名へ更新（古い @diar:N の復活防止）。
- 既存モード（Soniox単独、pyannote単独）は `cluster_namer is None` 分岐で完全に従来コード。
- テスト13件追加（test_cluster_naming.py 7件、test_recv_loop_cluster_merge.py 6件、新規）。
  sandbox 検証で live スイート含む全テストパス。実機での `uv run pytest -q` は §4-1 で確認のこと。
