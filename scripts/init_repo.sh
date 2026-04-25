#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

git init -b main
# Pin the local commit author so the initial commit doesn't inherit
# whatever happens to be in the global git config.
git config user.name  "Liu Su"
git config user.email "liusully@gmail.com"
git add .
git commit -m "Initial release: BTS-scored debate (Prelec 2004 ↦ Khan 2024)"

echo ""
echo "=== Next manual steps ==="
echo "1. Create empty repo on GitHub: gh repo create truth-serum-debate --public --source=. --remote=origin"
echo "2. Push: git push -u origin main"
echo "3. Verify: open https://github.com/liusulldel/truth-serum-debate"
