#!/usr/bin/env bash
# Reproduces experiment 001 from the repo root (after building core/):
#   experiments/001-double-down/run.sh explore   # every arm, seeds 1-5
#   experiments/001-double-down/run.sh confirm   # chosen arms, fresh seeds 101-110
#
# Exploration picks the candidates; confirmation measures them on seeds that
# played no part in picking them, with 1M evaluation rounds per policy per
# seed, so choosing the best of many arms doesn't flatter the winner.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../core"

run() { # phase seeds seed-base eval-rounds arm...
  local phase=$1 seeds=$2 base=$3 rounds=$4; shift 4
  for arm in "$@"; do
    mkdir -p "$HERE/$phase/$arm"
    ./build/experiment --config "$HERE/arms/$arm.cfg" --seeds "$seeds" \
      --seed-base "$base" --eval-rounds "$rounds" --out "$HERE/$phase" \
      --name "$arm" 2>/dev/null | grep -v '^Output\|^Wrote\|^Config' \
      > "$HERE/$phase/$arm/table.txt"
    echo "$phase/$arm done"
  done
}

case "${1:-all}" in
  explore|all)
    arms=()
    for cfg in "$HERE"/arms/*.cfg; do arms+=("$(basename "$cfg" .cfg)"); done
    run explore 5 1 200000 "${arms[@]}" ;;&
  confirm|all)
    run confirm 10 101 1000000 baseline gamma-1 visit-alpha expected-sarsa \
      double-q $(cat "$HERE/candidates.txt") ;;
  explore) ;;
  *) echo "usage: $0 [explore|confirm|all]" >&2; exit 1 ;;
esac
