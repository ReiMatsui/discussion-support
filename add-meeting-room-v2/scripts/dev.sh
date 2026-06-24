#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEAKER_MODEL="off"
SPEAKER_RESULTS=""

usage() {
  cat <<'EOF'
Usage: ./scripts/dev.sh [options]

Options:
  --speaker-model off|speechbrain|wespeaker|both
  --speaker-results PATH
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --speaker-model)
      SPEAKER_MODEL="${2:-}"
      shift 2
      ;;
    --speaker-results)
      SPEAKER_RESULTS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SPEAKER_MODEL" in
  off)
    UV_EXTRA=()
    ;;
  speechbrain)
    UV_EXTRA=(--extra speaker-speechbrain)
    ;;
  wespeaker)
    UV_EXTRA=(--extra speaker-wespeaker)
    ;;
  both)
    UV_EXTRA=(--extra speaker-both)
    ;;
  *)
    echo "Invalid --speaker-model: $SPEAKER_MODEL" >&2
    usage >&2
    exit 2
    ;;
esac

cleanup() {
  trap - INT TERM EXIT
  kill 0
}

trap cleanup INT TERM EXIT

echo "Starting backend on http://localhost:8000"
echo "Speaker verification model: $SPEAKER_MODEL"
(
  cd "$ROOT_DIR/backend"
  export SPEAKER_VERIFICATION_BACKEND="$SPEAKER_MODEL"
  if [[ -n "$SPEAKER_RESULTS" ]]; then
    export SPEAKER_VERIFICATION_RESULTS_PATH="$SPEAKER_RESULTS"
  fi
  uv run "${UV_EXTRA[@]}" uvicorn app.main:app --host 0.0.0.0 --port 8000
) &

echo "Starting experimental frontend"
(
  cd "$ROOT_DIR/frontend"
  VITE_VIEWER_WS_URL="${VITE_VIEWER_WS_URL:-ws://localhost:8000/ws/viewer}" \
    npm run dev
) &

wait
