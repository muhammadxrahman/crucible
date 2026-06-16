#!/usr/bin/env bash
# Build the web UI to web/dist, which `mlxd serve` serves at /.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d web/node_modules ]; then
  npm --prefix web install
fi
npm --prefix web run build
echo "==> built web/dist"
