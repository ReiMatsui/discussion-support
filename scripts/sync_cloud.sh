#!/usr/bin/env bash
# クラウド作業分を取り込む（1コマンド）。
#
# 事情: このリポジトリの正本は手元の Mac にあり、クラウド側の作業は
# git bundle で運ばれてくる（push できる remote が無いため）。毎回
#   git fetch <bundle> → git merge
# を手で打つのが面倒なので、ここにまとめる。
#
# 使い方:
#   ./scripts/sync_cloud.sh              # data/runs/cloud.bundle から取り込む
#   ./scripts/sync_cloud.sh ~/Downloads/ds.bundle
#
# やること:
#   1. バンドルから refs/remotes/cloud/main を更新
#   2. 作業ツリーの変更を一時退避（未追跡ファイルも含む）
#   3. 早送りマージ（履歴が分岐していたら中止して知らせる）
#   4. 退避したものを戻す
set -euo pipefail

BUNDLE="${1:-data/runs/cloud.bundle}"

cd "$(git rev-parse --show-toplevel)"

if [ ! -f "$BUNDLE" ]; then
    echo "バンドルが見つかりません: $BUNDLE" >&2
    exit 1
fi

echo "# 取り込み元: $BUNDLE"
echo "#   更新日時: $(date -r "$BUNDLE" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '不明')"
git fetch "$BUNDLE" main:refs/remotes/cloud/main --force

# バンドルの先端を必ず見せる。既定のパスに古いバンドルが残っていると
# 「Already up to date」だけが出て、取り込めたのか古いのか分からない
# （2026-07-28 に実際に起きた）。
echo "# バンドルの先端: $(git log --oneline -1 refs/remotes/cloud/main)"
if [ "$(git rev-parse HEAD)" = "$(git rev-parse refs/remotes/cloud/main)" ]; then
    echo "# 取り込むものはありません（手元は既にこの先端です）。"
    echo "#   新しい作業を待っているなら、このバンドル自体が古い可能性があります。"
    exit 0
fi

if git merge-base --is-ancestor HEAD refs/remotes/cloud/main; then
    :
else
    echo "履歴が分岐しています。早送りできません。" >&2
    echo "  手元:  $(git log --oneline -1 HEAD)" >&2
    echo "  雲側:  $(git log --oneline -1 refs/remotes/cloud/main)" >&2
    exit 1
fi

# 退避（何も無ければ stash は作られないので、その旨を覚えておく）
STASHED=no
if [ -n "$(git status --porcelain)" ]; then
    git stash push --include-untracked --quiet --message "sync_cloud: 自動退避"
    STASHED=yes
    echo "# 作業中の変更を退避しました"
fi

git merge --ff-only refs/remotes/cloud/main

if [ "$STASHED" = yes ]; then
    if git stash pop --quiet; then
        echo "# 退避した変更を戻しました"
    else
        echo "退避分を戻せませんでした。'git stash list' を確認してください。" >&2
        exit 1
    fi
fi

echo "# 完了: $(git log --oneline -1)"
