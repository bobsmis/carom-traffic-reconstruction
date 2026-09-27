#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:/usr/bin:/bin"

cleanup() {
  kill "${backend_pid:-}" "${frontend_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$project_root/backend" && exec "$project_root/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000) &
backend_pid=$!
(cd "$project_root/frontend" && exec npm run dev -- --host 0.0.0.0) &
frontend_pid=$!

printf 'Frontend: http://localhost:5173\n'
printf 'Backend:  http://localhost:8000/docs\n'
wait -n "$backend_pid" "$frontend_pid"
