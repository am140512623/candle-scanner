#!/usr/bin/env bash
# One-command YEAR-END review. Pulls results.csv (from the data repo's main branch)
# and every archived chart (from the `charts` branch), then builds gallery.html --
# each signal shown as its picture next to its data, with Keep/Delete toggles and a
# live-recomputing win-rate/net summary. Nothing is modified on the server.
#
# Requires: python (with this repo's deps) + a DATA_TOKEN with READ access to the
# private data repo. Run it from the repo root:  DATA_TOKEN=xxxx bash review.sh
#
# Output: ./review/gallery.html  (open it in any browser -- pics render offline).
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_REPO="${DATA_REPO:-am140512623/candle-data}"
CHARTS_BRANCH="${CHARTS_BRANCH:-charts}"
DIR="${REVIEW_DIR:-review}"

if [ -z "${DATA_TOKEN:-}" ]; then
  echo "Set DATA_TOKEN to a PAT with read access to ${DATA_REPO}."; exit 1
fi
URL="${DATA_REMOTE_URL:-https://x-access-token:${DATA_TOKEN}@github.com/${DATA_REPO}.git}"

rm -rf "$DIR"; mkdir -p "$DIR"

echo "Pulling results.csv (main)..."
git clone -q --depth 1 "$URL" "$DIR/_main"
cp "$DIR/_main/results.csv" "$DIR/results.csv"

echo "Pulling archived charts (${CHARTS_BRANCH} branch)... this can be large."
if git clone -q --depth 1 --branch "$CHARTS_BRANCH" "$URL" "$DIR/_charts" 2>/dev/null; then
  cp -r "$DIR/_charts/charts" "$DIR/charts"
else
  echo "  (no ${CHARTS_BRANCH} branch yet -- charts will show as 'not found')"
fi

echo "Building gallery..."
( cd "$DIR" && python "$ROOT/build_gallery.py" results.csv gallery.html )

echo
echo "Done. Open ${DIR}/gallery.html in your browser."
