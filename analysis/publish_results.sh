#!/usr/bin/env bash
# Regenerate every published artifact from ONE seeded run of config/tuned.cfg:
#   docs/training_report.txt, docs/images/*.png
#
# The README's figures must all come from the same run. Doing it by hand made
# it easy to refresh the plots and forget the report, or re-take a terminal
# screenshot from a different model; this script does all of it in one go.
#
# Usage (from the repo root, after building core/):
#   analysis/publish_results.sh [config]      # default: config/tuned.cfg
# Needs: core/build/{train,play}, python with analysis/requirements.txt, and
# `script` (util-linux) to give the binaries a pseudo-terminal for colour.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$(realpath "${1:-$ROOT/config/tuned.cfg}")"
PYTHON="${PYTHON:-$( [ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || echo python3 )}"
BUILD="$ROOT/core/build"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cd "$WORK"
echo "Training $CONFIG in $WORK ..."
"$BUILD/train" --config "$CONFIG" > train.log

cp analysis/training_report.txt "$ROOT/docs/training_report.txt"

"$PYTHON" "$ROOT/analysis/plot_training.py" logs/training_*.csv \
    --qtable analysis/q_table.csv --output "$ROOT/docs/images/"

# The colour chart and the AI hand come from the binaries' own terminal output.
script -qc "$BUILD/play --report --model ./models/final_agent" /dev/null \
    | "$PYTHON" "$ROOT/analysis/ansi_to_png.py" --until "^Not scored|^Divergences" \
        -o "$ROOT/docs/images/strategy_chart.png"
# Seed 59 deals split aces (one card each), a soft 20 that loses to a 21, and
# an 11 doubled against a queen: splitting, doubling and a loss in three hands.
"$BUILD/play" --mode ai --model ./models/final_agent --hands 3 --seed 59 \
    | "$PYTHON" "$ROOT/analysis/ansi_to_png.py" \
        -o "$ROOT/docs/images/play_ai_mode.png"

echo
grep -A6 "Expected Value" analysis/training_report.txt
grep -A3 "Strategy accuracy :" analysis/training_report.txt
echo "Published from $CONFIG"
