#!/bin/bash
# Live Transcriber 起動スクリプト
# ダブルクリックで起動 → ローカルサーバーを立ててChromeで開く

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PORT=8765
URL="http://localhost:${PORT}/live_transcriber.html"

# 既に同じポートで動いているプロセスを止める
if lsof -ti:${PORT} >/dev/null 2>&1; then
  echo "ポート ${PORT} を使用中のプロセスを停止します..."
  lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# Python 3 でサーバーを起動 (バックグラウンド)
echo "ローカルサーバーを起動中: ${URL}"
python3 -m http.server ${PORT} >/dev/null 2>&1 &
SERVER_PID=$!

# 終了時にサーバーを止める
trap "echo ''; echo 'サーバーを停止します...'; kill ${SERVER_PID} 2>/dev/null; exit 0" INT TERM

sleep 1

# Chromeで開く (なければデフォルトブラウザ)
if [ -d "/Applications/Google Chrome.app" ]; then
  open -a "Google Chrome" "${URL}"
else
  open "${URL}"
fi

echo ""
echo "======================================================"
echo "  Live Transcriber が起動しました"
echo "  URL: ${URL}"
echo ""
echo "  ブラウザでマイクの許可を求められたら「許可」を選択。"
echo "  終了するには、このウィンドウで Ctrl+C を押すか、"
echo "  ウィンドウを閉じてください。"
echo "======================================================"
echo ""

# サーバープロセスを待機 (Ctrl+Cで終了)
wait ${SERVER_PID}
