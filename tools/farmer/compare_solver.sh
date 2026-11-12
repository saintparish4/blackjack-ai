#!/usr/bin/env bash
# Diffs ./build/solve against Farmer's CA for one set of rules, block by block:
# the two-card strategy table, the EV by upcard, and the dealer table.
#
#   tools/farmer/compare_solver.sh DECKS H17 DAS SURRENDER     (run from core/)
#
# Prints "identical" per block, or the diff. Exits non-zero on any difference.
set -euo pipefail

if [ $# -ne 4 ]; then
  sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 1
fi

DECKS=$1 H17=$2 DAS=$3 SURRENDER=$4
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

FLAGS=(--decks "$DECKS" --any-ten-split)
[ "$H17" = y ] && FLAGS+=(--h17)
[ "$DAS" = y ] && FLAGS+=(--das)
[ "$SURRENDER" = y ] && FLAGS+=(--surrender)

./build/solve "${FLAGS[@]}" > "$TMP/ours.txt"
bash "$HERE/run_farmer.sh" "$DECKS" "$H17" "$DAS" "$SURRENDER" "$TMP/farmer.txt" > /dev/null

block() { # file start-pattern end-pattern
  sed -n "/$2/,/$3/p" "$1" | sed 's/ *+\/-.*//; s/ *$//'
}

status=0
check() { # name start end
  if diff <(block "$TMP/farmer.txt" "$2" "$3") <(block "$TMP/ours.txt" "$2" "$3") > "$TMP/d"; then
    echo "$1: identical ($(block "$TMP/ours.txt" "$2" "$3" | wc -l) lines)"
  else
    echo "$1: DIFFERENT"; cat "$TMP/d"; status=1
  fi
}
check "strategy table" '^  Hard |' '^  2- 2'
check "EV by upcard  " 'Overall expected value' '^  Total'
check "dealer table  " 'Probability of outcome' '^  Total'
exit $status