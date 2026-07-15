#!/usr/bin/env bash
# Best-effort archive of THIS run's chart PNGs to a dedicated `charts` branch of the
# PRIVATE data repo, so the `chart` links recorded in signals.csv resolve when you
# review results later.
#
# Design guarantees (this is crucial data, so it's built to be safe):
#   * ISOLATED  -- runs in its OWN clone dir, never touches the main-branch clone
#                  ($DATA_DIR) that pull_data/commit_data use for the CSVs.
#   * ADD-ONLY  -- it copies new PNGs in and commits; it never deletes, so a bug
#                  here can't wipe previously-archived charts.
#   * NON-FATAL -- it ALWAYS exits 0. If anything fails, CSV persistence and every
#                  bot carry on untouched; the only cost is that run's pics aren't
#                  saved.
#   * LEAN HOT PATH -- charts live on a SEPARATE branch, so the bots' pull_data.sh
#                  (which clones `main`) never downloads them. Only the year-end
#                  review pulls this branch.
#
# Requires env DATA_TOKEN (PAT with contents:write on the data repo).
# Optional  env DATA_REPO (default am140512623/candle-data), CHARTS_BRANCH (charts).
set -u
DATA_REPO="${DATA_REPO:-am140512623/candle-data}"
CHARTS_BRANCH="${CHARTS_BRANCH:-charts}"
SRC="${GITHUB_WORKSPACE:-.}/charts"
CDIR="${CHARTS_CLONE_DIR:-.charts-data}"

[ -n "${DATA_TOKEN:-}" ] || { echo "charts: DATA_TOKEN unset -- skipping archive."; exit 0; }
[ -d "$SRC" ] || { echo "charts: no charts/ dir -- nothing to archive."; exit 0; }
if ! find "$SRC" -type f -name '*.png' -print -quit 2>/dev/null | grep -q .; then
  echo "charts: no PNGs this run -- nothing to archive."; exit 0
fi

# DATA_REMOTE_URL overrides the remote (used by tests to point at a local repo);
# in production it's built from the token + repo as usual.
URL="${DATA_REMOTE_URL:-https://x-access-token:${DATA_TOKEN}@github.com/${DATA_REPO}.git}"
rm -rf "$CDIR"

# Shallow (depth 1) single-branch clone: brings the CURRENT charts (so we only ADD,
# never drop them) WITHOUT the branch's full history. If the branch doesn't exist
# yet, start a fresh orphan branch instead.
if git clone --quiet --depth 1 --single-branch --branch "$CHARTS_BRANCH" "$URL" "$CDIR" 2>/dev/null; then
  FRESH=0
else
  echo "charts: '$CHARTS_BRANCH' branch not found -- creating it."
  rm -rf "$CDIR"; mkdir -p "$CDIR"
  ( cd "$CDIR" && git init -q && git checkout -q --orphan "$CHARTS_BRANCH" \
      && git remote add origin "$URL" ) || { echo "charts: init failed -- skip."; exit 0; }
  FRESH=1
fi

# Merge this run's charts in (add-only; anything already there is left as-is).
mkdir -p "$CDIR/charts"
cp -r "$SRC/." "$CDIR/charts/" 2>/dev/null || true

cd "$CDIR" || exit 0
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add charts >/dev/null 2>&1 || true
if git diff --cached --quiet 2>/dev/null; then
  echo "charts: everything already archived -- nothing new."; exit 0
fi
N=$(git diff --cached --name-only | wc -l | tr -d ' ')
git commit -q -m "charts: archive ${N} file(s)" || { echo "charts: commit failed -- skip."; exit 0; }

# Push with retries -- several bots can finish and push around the same time. On a
# fresh branch the first push creates it; otherwise re-fetch + rebase so concurrent
# additions from other segments merge cleanly (different candles = different files).
for i in 1 2 3 4 5; do
  if [ "$FRESH" = "1" ]; then
    if git push -q origin "HEAD:${CHARTS_BRANCH}" 2>/dev/null; then
      echo "charts: pushed (new branch, ${N} file(s))."; exit 0
    fi
    FRESH=0   # push lost the create race -> branch now exists, switch to rebase path
  fi
  if git fetch -q --depth 1 origin "$CHARTS_BRANCH" 2>/dev/null \
     && git rebase -q "origin/${CHARTS_BRANCH}" 2>/dev/null \
     && git push -q origin "HEAD:${CHARTS_BRANCH}" 2>/dev/null; then
    echo "charts: pushed (${N} file(s))."; exit 0
  fi
  # Fallback in case the branch genuinely didn't exist yet.
  git push -q origin "HEAD:${CHARTS_BRANCH}" 2>/dev/null && { echo "charts: pushed (${N})."; exit 0; }
  echo "charts: push race -- retry ${i}..."; sleep $(( (RANDOM % 5) + 1 ))
done
echo "charts: could not push after retries (this run's pics not archived)."
exit 0
