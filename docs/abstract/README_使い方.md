# 中間試問会原稿のビルド環境（このフォルダの使い方）

## 最初に1回だけやること
1. MacTeX を入れる（無料・約6GB・30分ほど）
   - https://tug.org/mactex/ から MacTeX.pkg を落としてインストール
   - Homebrew 派なら: `brew install --cask mactex-no-gui`
   - インストール後、ターミナルを開き直す
2. VS Code に拡張「LaTeX Workshop」を入れる（⌘⇧X で LaTeX Workshop を検索）

## ふだんの使い方
1. VS Code で「このフォルダ」(docs/abstract) をフォルダごと開く
2. Matsui_Rei.tex を編集して ⌘S で保存 → 自動でビルドが走る
3. PDF は ⌘⌥V（または右上の虫めがねアイコン）で隣のタブに表示。以後は保存のたびに更新される

## 注意
- 初回ビルドは日本語フォントのキャッシュ作成で数分かかることがある。2回目以降は十数秒
- 提出はPDFのみ・ファイル名 Matsui_Rei.pdf・10MB以下・締切 9/7(月) 23:55（PandA + 指導教員・アドバイザへメール）
- ビルドに失敗するときは、ターミナルで `which latexmk` が通るか確認（通らなければMacTeXのインストール/再起動）
