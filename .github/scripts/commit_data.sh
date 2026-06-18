#!/usr/bin/env bash
# Commit data files (signals.csv / results.csv) back to the repo, so the
# otherwise-stateless bots gain a memory across runs. This is what lets us track
# what the price did after each alert.
#
# Usage:  commit_data.sh signals.csv [results.csv ...]
#
# Safe to call when nothing changed (it just no-ops), and it retries on push
# races -- several bots can finish around the same time and push to the same
# branch, so the first push may need a rebase before it lands.
set -u

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

git add "$@" 2>/dev/null || true
if git diff --cached --quiet; then
  echo "No data changes to commit."
  exit 0
fi

git commit -m "data: update $*" || exit 0

for i in 1 2 3 4 5; do
  if git pull --rebase --autostash origin "${GITHUB_REF_NAME}" && git push; then
    echo "Pushed data update."
    exit 0
  fi
  echo "Push race -- retry ${i}..."
  sleep $(( (RANDOM % 5) + 1 ))
done

echo "Could not push data after retries (next run will catch up)."
