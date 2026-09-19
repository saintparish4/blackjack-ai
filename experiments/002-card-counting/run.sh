#!/usr/bin/env bash
# Reproduces experiment 002 from the repo root (after building core/):
#   experiments/002-card-counting/run.sh explore   # every arm, seeds 1-5
#   experiments/002-card-counting/run.sh confirm   # pre-registered arms, fresh seeds 101-110
#
# Every arm also measures flat basic strategy and textbook Hi-Lo (with and
# without the Illustrious 18) on the same rules and shoes, so each folder
# holds the whole comparison at its penetration.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../core"

run() { # phase seeds seed-base arm...
  local phase=$1 seeds=$2 base=$3; shift 3
  for arm in "$@"; do
    mkdir -p "$HERE/$phase/$arm"
    ./build/experiment --config "$HERE/arms/$arm.cfg" --seeds "$seeds" \
      --seed-base "$base" --out "$HERE/$phase" --name "$arm" 2>/dev/null \
      | grep -v '^Output\|^Wrote\|^Config' > "$HERE/$phase/$arm/table.txt"
    echo "$phase/$arm done"
  done
}

case "${1:-all}" in
  explore|all)
    arms=()
    for cfg in "$HERE"/arms/*.cfg; do arms+=("$(basename "$cfg" .cfg)"); done
    run explore 5 1 "${arms[@]}" ;;&
  confirm|all)
    run confirm 10 101 $(cat "$HERE/candidates.txt") ;;
  explore) ;;
  *) echo "usage: $0 [explore|confirm|all]" >&2; exit 1 ;;
esac
