#!/usr/bin/env bash
# Fetch the result/state files from the PRIVATE data repo into the workspace, so
# the (public) scanners see their prior state before they run. Keeps results OUT
# of the public code repo -- the CSVs live only in the private repo.
#
# Requires env DATA_TOKEN: a PAT with contents:write on the data repo.
# Optional  env DATA_REPO (default am140512623/candle-data) and DATA_DIR (.data).
#
# Fail-safe: if DATA_TOKEN is missing or the clone fails, the scan still runs --
# it just starts from empty state (commit_data.sh will also no-op).
set -u
DATA_REPO="${DATA_REPO:-am140512623/candle-data}"
DATA_DIR="${DATA_DIR:-.data}"

if [ -z "${DATA_TOKEN:-}" ]; then
  echo "WARNING: DATA_TOKEN not set -- skipping private-data pull (empty state)."
  exit 0
fi

rm -rf "$DATA_DIR"
if git clone --quiet "https://x-access-token:${DATA_TOKEN}@github.com/${DATA_REPO}.git" "$DATA_DIR"; then
  for f in signals.csv results.csv summary.csv .index_signals_seen.txt; do
    [ -e "$DATA_DIR/$f" ] && cp "$DATA_DIR/$f" "./$f"
  done
  echo "Pulled data files from ${DATA_REPO}."
else
  echo "WARNING: could not clone ${DATA_REPO} -- scan will start from empty state."
fi
