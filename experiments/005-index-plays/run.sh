#!/usr/bin/env bash
# Reproduces experiment 005 from the repo root (after building core/):
#   experiments/005-index-plays/run.sh theory    # the EoR predictions
#   experiments/005-index-plays/run.sh explore   # every arm, seeds 1-5
#   experiments/005-index-plays/run.sh confirm   # candidates.txt, seeds 101-110
#
# The confirmation arms are pre-registered in candidates.txt, committed before
# confirm runs, so picking the best explorer can't flatter it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../core"
CHECKPOINTS=1e6,3e6,1e7,3e7,1e8,3e8

run() { # phase seeds seed-base arm...
  local phase=$1 seeds=$2 base=$3; shift 3
  for arm in "$@"; do
    mkdir -p "$HERE/$phase/$arm"
    ./build/indices --config "$HERE/arms/$arm.cfg" --seeds "$seeds" --seed-base "$base" \
      --checkpoints "$CHECKPOINTS" --out "$HERE/$phase" --name "$arm" \
      2>"$HERE/$phase/$arm/time.txt" | grep -v '^Wrote' > "$HERE/$phase/$arm/table.txt"
    echo "$phase/$arm done ($(cat "$HERE/$phase/$arm/time.txt"))"
  done
}

case "${1:-all}" in
  theory)
    ./build/indices --config "$HERE/arms/table.cfg" --theory > "$HERE/theory.txt"
    cat "$HERE/theory.txt" ;;
  explore|all)
    arms=()
    for cfg in "$HERE"/arms/*.cfg; do arms+=("$(basename "$cfg" .cfg)"); done
    run explore 5 1 "${arms[@]}" ;;&
  confirm|all)
    run confirm 10 101 $(cat "$HERE/candidates.txt") ;;
  explore) ;;
  *) echo "usage: $0 [theory|explore|confirm|all]" >&2; exit 1 ;;
esac
