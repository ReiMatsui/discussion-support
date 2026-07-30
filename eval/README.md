# eval/ の案内

数字の正本は `docs/design/STATUS.md`、経緯は handoff。新しい実験スクリプトは
**必ず `_pipeline.py` を呼ぶ**こと（再現を書き写すと必ずずれる。§29・§34）。
役目を終えたスクリプトは、結論を handoff に記録したうえで**消す**（git に
履歴が残る。§36・§44 で実施済み）。

## 共有ライブラリ（他が import する。消さない）

| ファイル | 役目 |
|---|---|
| `_pipeline.py` | **再現と採点の唯一の実装**。replay_seats / apply_schedule / score |
| `_gtlib.py` | GT読み込み・最適1:1対応 |
| `_textgt.py` | 正解の文章一致割当て（§34） |
| `_utt_embeddings.py` | 発話ごとの声紋の計算と保存（`_emb/`、b5） |
| `decompose_attribution.py` | diag の読み込み（load_run）・相槌判定 |

## 現役の測定スクリプト

| ファイル | 何を測るか |
|---|---|
| `error_anatomy.py` | 誤りの内訳（長さ・経路・またぎ/重なり別）。§35/§37 |
| `gt_alignment_compare.py` | 正解の当て方（時間/文章）の比較。§34 の再現 |
| `short_utterance_pick.py` | 短い発話の棄権則の校正。§36 |
| `embedding_model_compare.py` | 埋め込みモデル比較（b2/b3/b5/b6）。§38 |
| `roster_dependency.py` | 人数の有無の効果（§44 の訂正済みラベル） |
| `roster_free.py` | 人数なしのクラスタリング上限（§40） |
| `roster_merge.py` | キー統合案（却下の証拠。§44.3） |
| `method_overlap.py` | 本番と単純方式の発話ごとの突き合わせ。§41 |
| `segment_split_ceiling.py` | 区切り直しの上限（却下の証拠。§37） |
| `retro_schedule.py` | 遡及訂正の予定表の校正。§28 |
| `replay_seat_assign.py` | 本番の SeatAudio を記録で駆動する忠実性確認。§27 |

## 運用ツール

| ファイル | 役目 |
|---|---|
| `annotate.py` (+`_annot_html.py`) | 音声を聴いて GT を付ける画面 |
| `watch.py` | 実走ランを音声つきで再生して見る画面 |
| `diagnose_live_session.py` | 実走ランの一次診断 |
| `transplant_gt.py` | タイムラインGTを別ランへ移植 |
| `run_chiba.py` / `prep_chiba.py` / `prep_sakura.py` | コーパスの再生と準備 |

## 旧世代（結論は handoff に記録済み。新規の測定には使わない）

`replay_attribution.py`（声紋層の再設計の実測記録。src/tests の docstring から
多数引用されるため残す） / `replay_live_attribution.py`（§23。diagnose が使う） /
`eval_speaker_gt.py`（run_chiba が採点に使う） / `score_transcription.py`

削除済み（`git log --diff-filter=D -- eval/<名前>.py` で履歴から復元できる）:
`seat_query_context` `seat_pick_variants` `seat_assign_extensions`
`retro_reattribution`（§27〜§28） / `cluster_merge_feasibility`
`phase0_dual_ledger`（二重帳簿 handoff に記録） / `score_pyannote_ceiling`
`fetch_callhome_jpn`（§14期） / `segment_boundary_ceiling`（§37で置換） /
`sortformer_compare` `sortformer_infer`（§29で却下）
