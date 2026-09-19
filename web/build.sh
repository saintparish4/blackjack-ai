#!/usr/bin/env bash
# Build the WebAssembly module for the browser demo into web/wasm/.
#
# Needs Emscripten: emcc on PATH, or an emsdk at $EMSDK or ~/emsdk.
# Then preview with:  python3 -m http.server -d web 8000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v emcc >/dev/null 2>&1; then
  for env in "${EMSDK:-}/emsdk_env.sh" "$HOME/emsdk/emsdk_env.sh"; do
    if [ -f "$env" ]; then
      # shellcheck disable=SC1090
      source "$env" >/dev/null 2>&1
      break
    fi
  done
fi
command -v emcc >/dev/null 2>&1 || {
  echo "emcc not found. Install Emscripten: https://emscripten.org/docs/getting_started/" >&2
  exit 1
}

emcmake cmake -B "$ROOT/build-wasm" -S "$ROOT/core" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/build-wasm" --parallel --target blackjack_wasm
mkdir -p "$ROOT/web/wasm"
cp "$ROOT/build-wasm/blackjack.mjs" "$ROOT/build-wasm/blackjack.wasm" "$ROOT/web/wasm/"
echo "Built web/wasm/. Preview: python3 -m http.server -d web 8000"
