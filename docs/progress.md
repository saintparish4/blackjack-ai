# Progress and metrics

One page for checking where the project stands. Each step appends a milestone below and updates the
scoreboard. Every number here is reproducible from a config file plus a seed, and is measured
through the same `Evaluator`, game engine and rules as the in-engine baselines.

## Scoreboard

_Last updated 2026-09-19, after step 4 (card counting). All four next steps have landed._

| Area | Metric | Value | Reproduce with (from `core/`) |
|------|--------|-------|-------------------------------|
| **Headline** | Learned agent EV per round | **−0.0098 ± 0.0049** | `./build/train --config ../config/tuned.cfg` |
| | Basic strategy EV per round (same run) | −0.0052 ± 0.0050 | same run |
| | Random play EV per round (same run) | −0.4443 ± 0.0057 | same run |
| | EV gap recovered | **99.0%** | same run |
| **Confirmed** | Agent vs book, 10 fresh seeds × 1M rounds | −0.0071 ± 0.0008 vs −0.0054 ± 0.0006 | `experiments/001-double-down/run.sh confirm` |
| | Shortfall to basic strategy | **0.00112 ± 0.00029 per round, exact** (was 0.0248; 001's Monte Carlo said 0.0017) | `experiments/004-regret/` |
| **Spread** | EV gap recovered, seeds 1–5 | 99.7% ± 1.0 (range 98.8–100.6) | `./build/experiment --config ../config/tuned.cfg --seeds 5` |
| | Agent EV, seeds 1–5 | −0.0076 ± 0.0024 (range −0.0098 to −0.0049) | same |
| **Policy** | Accuracy: overall / two-card / multi-card / pairs | 94.9% / 92.9% / 97.8% / 92.0% | seed 1 |
| | Book doubles (hard 9–11) the agent makes | 18 of 21 (was 3) | seed 1 strategy chart |
| **Engine** | Basic strategy EV, 20M rounds | −0.0061 ± 0.0004 (published house edge for these rules: ≈ −0.005 to −0.006) | `./build/experiment --config ../config/tuned.cfg --seeds 4 --episodes 1000 --eval-rounds 5000000` |
| **Engineering** | Unit tests | 166 passing, 10 suites | `./build/run_tests` |
| | Training throughput | 1.1M episodes/s in the published run (machine shared with other jobs; 2.0–2.7M/s measured idle) | training report |
| | Game simulation | 5.6–6.1M rounds/s (step 1 measurement) | `./build/benchmark` |
| | Greedy decision latency | 3.9–4.5 ns (step 1 measurement) | `./build/benchmark` |
| **Lab** | Experiments recorded | 3 ([`000`](../experiments/000-tuned-baseline/), [`001`](../experiments/001-double-down/), [`002`](../experiments/002-card-counting/)) | [`experiments/`](../experiments/) |
| | Arms run | 001: 18 explored, 7 confirmed · 002: 10 explored, 4 confirmed | experiment folders |
| **Counting** | Basic strategy + learned bets, edge per unit wagered | **+0.27% ± 0.05** (10 fresh seeds; flat basic strategy −0.58%) | `experiments/002-card-counting/run.sh confirm` |
| | Textbook Hi-Lo ramp / Hi-Lo + Illustrious 18 (same shoes) | +0.11% ± 0.04 / +0.22% ± 0.03 | same |
| | Units won per 100 rounds: learned bets vs Hi-Lo + I18 | +0.74 ± 0.14 vs +0.32 ± 0.04 | same |
| | Desirability index: learned bets vs Hi-Lo + I18 vs Hi-Lo ramp | 1.62 vs 1.41 vs 0.71 | same |
| | Learned counter (count in the playing state, 50M episodes) | −0.10% ± 0.13, a negative result | same |
| | Agent's measured edge per true count | about +0.5% per count, zero near TC +1 | `count_profile.csv` |
| **Product** | Browser demo | built and tested; goes live at https://saintparish4.github.io/blackjack-ai/ once pushed with Pages enabled | `web/build.sh`, then `python3 -m http.server -d web` |
| | Demo payload | 261 KB wasm + 102 KB JS + 43 KB model | `web/build.sh` |
| | Browser build vs C++ | 20,000 rounds at −0.0117/round (C++ evaluator: −0.0098 ± 0.0049); 221,749 smoke checks | `node web/test/smoke.mjs` |

`±` is a 95% confidence interval. For a single run it comes from the per-round standard error over
200,000 rounds. Across seeds it is a t-interval on the mean of the per-seed values.

## Roadmap status

- [x] **Step 1: lab bench.** Seeds, experiment runner, early stopping on accuracy or EV, payout fix.
- [x] **Step 2: teach it to double.** Visit-scaled α wins; the shortfall to the book is 0.0017/round.
- [x] **Step 3: browser demo.** WebAssembly build with Play, Coach and Watch modes, deployed by CI.
- [x] **Step 4: card counting.** Learned bet sizing beats textbook Hi-Lo's ramp; count-aware play
  didn't learn (a negative result, written up).

## Milestones

### Step 4: card counting (2026-09-19)

**What landed**

- **Engine:** a Hi-Lo running count of the cards the player has seen. The hole card is counted only
  once revealed. There's a true count over unseen decks, a pre-deal reshuffle hook for betting, and
  a table bet spread (`max_bet`).
- **State and model:**
  - `State` gains a true-count bucket: floor(TC) clamped to −3…+6, with 0 for count-blind agents,
    whose hashes don't change.
  - `PolicyTable` widens to 65,536 slots on the heap.
  - `.qtable` v3 (v1 and v2 still load).
  - The CSV export gains `count_bucket`.
- **Agents:** optional counting hooks on `Agent`. `TabularAgent` gets `use_count` and `learn_bets`
  (with max-when-positive or proportional sizing). There's also a basic-strategy bettor for
  ablation.
- **Baselines:** `HiLoAgent` with a 1–8 ramp, optionally plus the Illustrious 18 (Schlesinger's
  S17 indices, without insurance).
- **Evaluator:** bets at the pre-deal count, and reports:
  - EV per unit wagered with a delta-method CI
  - units won per 100 rounds
  - the SD of winnings and the desirability index
  - an edge-by-count profile
- **Experiment 002:** pre-registered in git before confirmation. 10 arms explored and 4 confirmed
  on fresh seeds, with penetration at 50/75/90%.
- CI runs a short counting experiment twice and diffs it. Non-counting results reproduce byte for
  byte.

**Metrics at this milestone** (confirmation: seeds 101–110, 10M rounds per policy per seed)

| Policy | EV / unit wagered | Units / 100 rounds | DI |
|---|---|---|---|
| Basic strategy + learned bets | +0.27% ± 0.05 | +0.74 ± 0.14 | 1.62 ± 0.31 |
| Hi-Lo + Illustrious 18 | +0.22% ± 0.03 | +0.32 ± 0.04 | 1.41 ± 0.18 |
| Learned play + learned bets | +0.16% ± 0.07 | +0.40 ± 0.19 | 0.96 ± 0.42 |
| Hi-Lo ramp | +0.11% ± 0.04 | +0.16 ± 0.07 | 0.71 ± 0.29 |
| Learned counter (count in state) | −0.10% ± 0.13 | −0.18 ± 0.22 | −0.55 ± 0.74 |
| Flat basic strategy | −0.58% ± 0.02 | −0.58 | n/a |

| Penetration (explore, seeds 1–5) | Learned bettor | Hi-Lo ramp | Hi-Lo + I18 |
|---|---|---|---|
| 50% | −0.08% | −0.36% | −0.25% |
| 75% | +0.16% | +0.05% | +0.26% |
| 90% | +0.33% | +0.42% | +0.70% |

**Honest limits:**

- The learned agent doesn't learn index plays; counting helps it only through bets.
- Its best bet sizing is "maximum whenever the edge is positive". That wins more per round than
  the textbook ramp, but at more risk; on the risk-adjusted DI it ties Hi-Lo + I18 rather than
  beating it.
- No bankroll or risk-of-ruin modelling yet.

### Step 3: browser demo (2026-09-19)

**What landed**

- **`web/`:** a no-build page with three modes.
  - **Play** grades each of your moves against the agent's Q-values: "the AI would hit; by its
    estimate that cost you about 0.19 of a bet".
  - **Coach** shows the agent's move, Q-value bars, confidence and a plain-English reason before
    you act.
  - **Watch the AI** steps through its decisions or auto-plays.
  - Also: a 10-hand scorecard, and `?seed=N` to replay a shoe.
- **`core/bindings/wasm/`:** the same engine, published model and basic strategy chart, compiled
  with Emscripten. The model is the published `tuned.cfg` run, byte for byte.
- **`ai/Explain.hpp`:** one source for the plain-English reasons, shared with `play --beginner`.
- **CI (`pages.yml`):** builds the module with a pinned Emscripten, runs a Node smoke test through
  the page's own API, and deploys to GitHub Pages from master.
- Checked in headless Chromium at desktop and mobile sizes, with no console errors. Two CSS bugs
  found that way were fixed: controls meant for another mode showing, and a blank selected tab.

**Metrics at this milestone**

| Metric | Value |
|--------|-------|
| First-visit download | ~406 KB (wasm 261 KB, JS 102 KB, model 43 KB) before compression |
| Browser build EV, 20,000 rounds | −0.0117 per round (C++: −0.0098 ± 0.0049 on 200k) |
| AI agrees with the book on live decisions | 94.9% of 26,947 |
| Smoke-test checks | 221,749 |
| Tests (C++) | 152 |

**Not verifiable from here:** the public URL goes live only after a push to GitHub with Pages
enabled (Settings → Pages → Source: GitHub Actions). Visitor analytics, the step's startup probe,
aren't wired up yet.

### Step 2: teach it to double (2026-09-19)

**What landed**

- **Visit-scaled step size** (`alpha_schedule = visit`, α = 1/(1+n)^ω), now the default in both
  configs. Also new: `ExpectedSarsaAgent`, `DoubleQLearningAgent`, and a `TabularAgent` base they
  share.
- **Engine:** split aces take one card each, and 21 after a split is no longer a natural paying 3:2.
- **Yardstick:** full basic strategy for the engine's rules, with soft doubles, double-else-stand,
  no-DAS pair splits and S17/H17 differences, and surrender only on opening hands.
- **Split credit:** SPLIT is valued from both hands, γ·(V₁ + V₂), and each hand's last decision
  learns its own result.
- **Audit:** pair decisions are now scored and charted.
- **Experiment 001:** 18 arms explored on seeds 1–5, 7 confirmed on fresh seeds 101–110.

**Metrics at this milestone**

| Metric | Step 1 (old engine and chart) | Step 2 (seed 1) |
|--------|--------------------------------|-----------------|
| Agent EV per round | −0.0294 ± 0.0045 | −0.0098 ± 0.0049 |
| Basic strategy EV | −0.0023 ± 0.0049 (inflated by engine bugs) | −0.0052 ± 0.0050 |
| EV gap recovered | 93.9% | 99.0% |
| Agent vs book, confirmed | n/a | −0.0071 vs −0.0054 (10 seeds × 1M rounds) |
| Accuracy: two-card | 84.6% | 92.9% |
| Accuracy: pairs | not audited | 92.0% |
| Book doubles made (hard 9–11) | 3 | 18 of 21 |
| Tests | 131 | 150 |

The old and new book numbers aren't comparable. The old engine paid 3:2 on A+T after a split and
let split aces draw, which flattered every policy. With the engine fix and the fuller chart together,
basic strategy went from −0.0033 (5 seeds) to −0.0061 (20M rounds), in line with the published house
edge for these rules.

### Step 1: lab bench (2026-09-19)

**What landed**

- `--seed` (and a `seed` config key) fixes the agent's exploration, the training shoe and every
  evaluation shoe. Streams are derived from one seed with splitmix64, so they don't move together.
- `./build/experiment` trains one config over N seeds in parallel and writes `results.csv` and
  `summary.csv`. Stdout and CSVs depend only on (config, seeds); timings go to stderr. CI runs it
  twice and diffs the output.
- Early stopping watches policy-audit accuracy (default) or EV, not win rate.
- `blackjack_payout` now reaches the reward. A 6:5 table used to train and score as if it paid 3:2.
- Report fixes: fractional durations, throughput that excludes evaluation time, 10 significant
  digits for hyperparameters (the old `0.999999` was really `0.9999985`), and outcome rates over
  200k rounds instead of the last 2,000-round evaluation.
- `extends = <path>` in config files, so an experiment arm is a small diff against `tuned.cfg`.
- `train --seed N` and seed N of an experiment report identical numbers for the identical agent.

**Metrics at this milestone**

| Metric | Before (2026-08-30, unseeded, `7163b4a`) | After (seed 1) |
|--------|----------------------|----------------|
| Agent EV per round | −0.0368 ± 0.0045 | −0.0294 ± 0.0045 |
| Basic strategy EV | −0.0034 ± 0.0049 | −0.0023 ± 0.0049 |
| EV gap recovered | 92.4% | 93.9% |
| Five-run spread of gap recovered | 91.5–93.1% (by hand) | 91.2–95.2% (`experiment --seeds 5`) |
| Accuracy: overall / two-card / multi-card | 86.2 / 81.2 / 90.4% | 87.3 / 84.6 / 89.6% |
| Tests | 114 | 131 |

The algorithm didn't change. The new numbers are a different draw from the same distribution, now
pinned to a seed, and the five-seed spread matches the hand-run spread the README quoted before.

## Metric definitions

- **EV per round**: mean reward per round against the initial bet. Win +1, blackjack +payout (1.5
  at 3:2), push 0, loss −1, surrender −0.5, doubled hands ×2, split hands summed.
- **EV gap recovered**: (agent − random) / (basic strategy − random), with all three measured on
  the same rules by the same evaluator. 0% plays like random, 100% like the book.
- **Strategy accuracy**: share of audited states where the greedy action matches basic strategy.
  It's scored separately for two-card hands (where doubling lives), multi-card hands, and opening
  pairs (since step 2). States where some legal action was never trained are reported as coverage
  gaps, not scored.
- **Shortfall to basic strategy**: book EV − agent EV per round. Unlike gap recovered, it doesn't
  shrink just because random play is terrible.
- **EV per unit wagered** (counting runs): units won ÷ units bet. Equal to EV per round for a flat
  bettor. Its CI comes from the delta method.
- **Desirability index (DI)**: 1000 × mean ÷ SD of per-round winnings (Schlesinger). A risk-adjusted
  score that doesn't depend on the unit size, so bettors with different spreads compare fairly.
- **Win rate** isn't tracked as progress. It sits near 43% for almost any sensible policy.
