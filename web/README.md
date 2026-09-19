# Browser demo

The C++ engine, the trained Q-table and the basic strategy chart, compiled to WebAssembly and
wrapped in a small page with no build step. Three modes:

- **Play:** you play, and after each move the AI says what it would have done and what the
  difference cost, by its own estimate.
- **Coach:** the AI shows its move, its Q-values and a plain-English reason before you decide.
- **Watch the AI:** step through its decisions or let it auto-play.

Nothing leaves the browser. `?seed=N` in the URL replays the same shoe.

## Build and run locally

```bash
web/build.sh                          # needs Emscripten (emcc, or an emsdk at ~/emsdk)
python3 -m http.server -d web 8000    # then open http://localhost:8000
node web/test/smoke.mjs               # 20,000 rounds through the page's API
```

## Layout

| Path | What |
|------|------|
| `../core/bindings/wasm/blackjack_wasm.cpp` | The `Table` class exposed to JS via Embind; returns JSON |
| `js/coach.mjs` | Pure coaching logic (grading, costs, session totals), shared with the Node test |
| `js/app.mjs` | DOM wiring |
| `model/` | The published model (`analysis/publish_results.sh` refreshes it) |
| `wasm/` | Build output, gitignored; CI rebuilds it for GitHub Pages |
| `test/smoke.mjs` | Node smoke test run in CI |
| `test/screenshots.cjs` | Optional: regenerates `docs/images/web_*.png` with Playwright |

The "cost" of a move is Q(AI's move) − Q(your move) from the agent's own table. It's an estimate,
and it's only shown when every legal action in that spot was trained. Where the agent and basic
strategy disagree, the panel says so.
