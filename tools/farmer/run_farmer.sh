#!/usr/bin/env bash
# Runs Eric Farmer's blackjack combinatorial analyzer, the external reference
# the solver is checked against, for one set of rules.
#
#   tools/farmer/run_farmer.sh DECKS H17 DAS SURRENDER OUT [HAND...]
#
# H17, DAS and SURRENDER are y or n. Rules not listed are fixed to what the
# engine plays: double on any two cards (soft included), no resplits, 3:2,
# CDZ- post-split strategy. Each HAND is Farmer's "up count card card..."
# query, e.g. "10 2 10 6" for 10,6 against a ten; the answers go to stdout.
#
#   tools/farmer/run_farmer.sh --build-only
#
# just fetches and builds, including bin/policy_ev (see policy_ev.cpp).
#
# Farmer's code is GPL-3.0 and this repo is MIT, so it is never vendored:
# it is cloned at a pinned commit into tools/farmer/src (gitignored) the
# first time this runs.
set -euo pipefail

if [ $# -lt 5 ] && [ "${1:-}" != "--build-only" ]; then
  sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/src"
BIN="$HERE/bin"
COMMIT=a1f7dbb74266fb39296292bdff568b076120a61c

if [ ! -x "$BIN/strategy" ]; then
  if [ ! -d "$SRC/.git" ]; then
    git clone --quiet https://github.com/possibly-wrong/blackjack.git "$SRC"
  fi
  git -C "$SRC" checkout --quiet "$COMMIT"
  mkdir -p "$BIN"
  # blackjack_split.cpp is left out: it carries its own main().
  g++ -O2 -w -I"$SRC/src" -I"$SRC/src/pdf" -o "$BIN/strategy" \
    "$SRC/src/strategy.cpp" "$SRC/src/blackjack.cpp" \
    "$SRC/src/pdf/blackjack_pdf.cpp" "$SRC/src/indices.cpp"
fi
if [ ! -x "$BIN/policy_ev" ] || [ "$HERE/policy_ev.cpp" -nt "$BIN/policy_ev" ]; then
  g++ -O2 -w -I"$SRC/src" -o "$BIN/policy_ev" "$HERE/policy_ev.cpp" \
    "$SRC/src/blackjack.cpp" "$SRC/src/indices.cpp"
fi
if [ "${1:-}" = "--build-only" ]; then exit 0; fi

DECKS=$1 H17=$2 DAS=$3 SURRENDER=$4 OUT=$5
shift 5

# Answers to strategy's prompts, in the order it asks them: decks, H17,
# double any total, double soft, double after hit, DAS, resplit, CDZ-,
# surrender, payout, output file; then hand queries, then "0 0" to quit.
{
  printf '%s\n' "$DECKS" "$H17" y y n "$DAS" n y "$SURRENDER" 1.5 "$OUT"
  for hand in "$@"; do printf '%s\n' "$hand"; done
  printf '0 0\n'
} | "$BIN/strategy" | grep "E(X)" || true