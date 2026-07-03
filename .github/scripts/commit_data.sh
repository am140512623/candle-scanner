#!/usr/bin/env bash
# Persist data files to the PRIVATE data repo, keeping results OUT of the public
# code repo. pull_data.sh must have cloned the data repo to $DATA_DIR first.
#
# Usage:  commit_data.sh signals.csv [results.csv ...]
#
# Requires env DATA_TOKEN (PAT with contents:write on the data repo).
# Safe to call when nothing changed (no-ops), and it retries on push races --
# several bots can finish around the same time and push to the same branch.
set -u
DATA_REPO="${DATA_REPO:-am140512623/candle-data}"
DATA_DIR="${DATA_DIR:-.data}"

if [ -z "${DATA_TOKEN:-}" ] || [ ! -d "$DATA_DIR/.git" ]; then
  echo "Private data repo not available (DATA_TOKEN unset or pull skipped) -- not persisting."
  exit 0
fi

# Copy the freshly-updated workspace files into the data-repo clone.
for f in "$@"; do
  [ -e "$f" ] && cp "$f" "$DATA_DIR/$f"
done

cd "$DATA_DIR" || exit 0
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git remote set-url origin "https://x-access-token:${DATA_TOKEN}@github.com/${DATA_REPO}.git"

git add "$@" 2>/dev/null || true
if git diff --cached --quiet; then
  echo "No data changes to commit."
  exit 0
fi

git commit -m "data: update $*" || exit 0

for i in 1 2 3 4 5; do
  if git pull --rebase --autostash origin main && git push origin HEAD:main; then
    echo "Pushed data update to ${DATA_REPO}."
    exit 0
  fi
  echo "Push race -- retry ${i}..."
  sleep $(( (RANDOM % 5) + 1 ))
done

echo "Could not push data after retries (next run will catch up)."
