# 設計書: ライブUIの /console・/board 分離と可視性改善

実施者向け。根拠: `docs/review-2026-07-02/04_ui_ux.md`（Critical: 接続断の無音故障 / High: AI発話停止手段なし・介入が目立たない / 本質提案: 操作卓と参加者表示の分離）、logic_review C17（モダリティ配分）。対面パイロット#2・段階Cで使う画面。手動確認は必要だがブラウザだけで可能（対面実験不要）。品質基準は共通。

## 方針

現行の単一ページ（ホスト用設定と議事録が混在、127.0.0.1バインド）を2ビューに分離する。**新規フレームワークは入れない**（現行の素のHTML/JS/SSE構成を維持。755行の埋め込みHTML文字列は最低限のファイル分離のみ行う）。

## U1 `refactor(live): HTMLテンプレートの別ファイル化`
`_webapp.py` 内の巨大文字列を `src/das/asr/live/web/console.html` / `board.html` / 共通 `live.js` `live.css` として分離し、配信ハンドラで読む（パッケージデータとして同梱、`importlib.resources` 経由）。挙動不変のリファクタ。既存UIテスト（test_ui_api）が通ること。

## U2 `feat(live): /board（参加者向け表示）新設`
- 内容: ライブ議事録（大きめ文字）＋**最新介入バナー**（宛先名入り・介入種別アイコン・30秒で自動フェード）＋議題/論点リスト。操作要素はゼロ
- 接続健全性: STT/エージェントの接続状態が snapshot に無いのが現状の Critical。`api_snapshot` に `health: {stt: ok|reconnecting|dead, agent: ok|reconnecting|off, last_confirm_sec}` を追加（`_bootstrap` の再接続ロジックと RealtimeAgent の `_connected` から取得）。board/console 双方でヘッダに常時表示し、**文字起こしが60秒止まったら明示の警告帯**
- バインド: `--host 0.0.0.0` オプション（既定は現行どおり localhost。会議室の別端末から /board を開く用途）。認証は付けない代わりに、0.0.0.0 時は起動ログに警告を出す

## U3 `feat(live): /console（ホスト操作卓）= 現行UIの整理`
- 現行ページを console と改名し、設定系（人数・声紋登録・積極性・モード・議題編集・リセット・停止）と診断系（発言量・triage/介入ステータス）を残す
- **AI発話の即時停止ボタン**を追加（High対応）: 既存 `agent.interrupt()` を叩く `/api/interrupt` を新設。「今の発話を止める」1タップ。停止イベントは interventions.jsonl に `cancelled_by_ui` として記録（受容性指標の副産物）
- 手動呼び出しボタンは console/board 両方に置く（board からの呼び出しは参加者の一次手段）

## U4 `fix(live): SSEの差分配信`（04章 High、長時間会議の劣化）
毎秒フルスナップショット＋innerHTML全置換をやめ、records は「前回送信以降の追加分＋rekey時のみ全量」を送る方式に変更。クライアントは追記描画。rev 機構は既にあるので、サーバ側で「差分 or 全量」を rev ギャップで判断。テキスト選択が保持されることを手動確認。テスト: 差分/全量切替のユニットテスト。

## 実施順序と確認
U1（挙動不変）→ U2（board+health）→ U3（console+interrupt）→ U4（差分）。各段でブラウザ手動確認（シミュレートモードで可）: 2ブラウザ同時表示、リセット・モード切替・切断（STTプロセスkill）時の表示。スクリーンショットをPR説明に添付。

## スコープ外
- 認証・HTTPS、モバイル最適化、pyvisグラフのライブ埋め込み（AF統合後に /board への「今の論点マップ」表示を別途検討——H1 の snapshot が出てから）
