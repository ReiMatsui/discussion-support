#!/bin/bash
# Live Transcriber (OpenAI Realtime API) 起動スクリプト
# ダブルクリックで起動 → Pythonサーバーを立ててChromeで開く

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PORT=8765
URL="http://localhost:${PORT}/"
KEY_FILE="${HOME}/.openai_api_key"
ENV_FILE="${DIR}/.env"

# ---- 1) Python 3 のチェック ----
if ! command -v python3 >/dev/null 2>&1; then
  echo ""
  echo "[error] python3 が見つかりません。"
  echo "        https://www.python.org/downloads/macos/ からインストールしてください。"
  echo ""
  read -n 1 -s -r -p "Enterキーで閉じます..."
  exit 1
fi

# ---- 2) APIキーのチェック / 設定 ----
if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f "${ENV_FILE}" ] && [ ! -f "${KEY_FILE}" ]; then
  echo ""
  echo "================================================="
  echo "  初回セットアップ: OpenAI API キー"
  echo "================================================="
  echo "  https://platform.openai.com/api-keys で取得した"
  echo "  sk-... で始まるキーを貼り付けてEnter:"
  echo ""
  read -r -s -p "API Key: " ENTERED_KEY
  echo ""
  if [ -z "${ENTERED_KEY}" ]; then
    echo "[error] キーが入力されませんでした。"
    read -n 1 -s -r -p "Enterキーで閉じます..."
    exit 1
  fi
  echo "${ENTERED_KEY}" > "${KEY_FILE}"
  chmod 600 "${KEY_FILE}"
  echo "[ok] キーを ${KEY_FILE} に保存しました (パーミッション 600)"
  echo ""
fi

# ---- 3) 既存サーバー停止 ----
if lsof -ti:${PORT} >/dev/null 2>&1; then
  echo "[info] ポート ${PORT} を使用中のプロセスを停止します..."
  lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# ---- 3.5) 仮想環境(venv)の準備と依存インストール ----
# システムのpythonに直接インストールすると環境によって import できない
# 問題が起きるため、専用のvenvを作ってそこに aiohttp を入れる。
VENV="${DIR}/.venv"
PYBIN="${VENV}/bin/python"

if [ ! -x "${PYBIN}" ]; then
  echo "[setup] 仮想環境を作成します (${VENV})..."
  python3 -m venv "${VENV}"
fi

# aiohttp が無ければインストール
if ! "${PYBIN}" -c "import aiohttp" >/dev/null 2>&1; then
  echo "[setup] aiohttp をインストールします..."
  "${PYBIN}" -m pip install --quiet --upgrade pip
  "${PYBIN}" -m pip install --quiet aiohttp
fi

# ---- 4) サーバー起動 ----
echo ""
echo "================================================="
echo "  Live Transcriber を起動します"
echo "  URL: ${URL}"
echo "================================================="
echo ""

# サーバーを起動 (フォアグラウンドで実行してログを表示)
"${PYBIN}" server.py &
SERVER_PID=$!

# サーバーが起動するまで待機
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -s "http://localhost:${PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

# Chromeで開く
if [ -d "/Applications/Google Chrome.app" ]; then
  open -a "Google Chrome" "${URL}"
else
  echo "[warn] Google Chrome が見つかりません。デフォルトブラウザで開きます。"
  open "${URL}"
fi

echo ""
echo "[info] このウィンドウを閉じるか Ctrl+C で停止します"
echo ""

# クリーンアップ
trap "echo ''; echo '[info] サーバーを停止します...'; kill ${SERVER_PID} 2>/dev/null; exit 0" INT TERM

wait ${SERVER_PID}
