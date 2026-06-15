#!/usr/bin/env bash
# Install the git pre-push hook that runs the regression gate before every push.
# Run once after cloning:  ./scripts/install-hooks.sh
set -euo pipefail

cd "$(dirname "$0")/.."
hook=".git/hooks/pre-push"

cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Block the push if the regression suite fails. Mac/Metal tests run locally by design.
set -euo pipefail
echo "pre-push: running regression gate (scripts/check.sh)"
exec ./scripts/check.sh
HOOK

chmod +x "$hook"
echo "installed $hook"
