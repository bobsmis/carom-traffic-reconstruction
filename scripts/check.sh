#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:/usr/bin:/bin"

cd "$project_root/frontend"
npm run build
npm run lint

cd "$project_root/backend"
PYTHONPATH=. "$project_root/.venv/bin/pytest" -q
