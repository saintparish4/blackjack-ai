# Progress and metrics

One page for checking where the project stands. Each step appends a milestone below and updates the
scoreboard. Every number here is reproducible from a config file plus a seed, and is measured
through the same `Evaluator`, game engine and rules as the in-engine baselines.

## Scoreboard

_Last updated 2026-09-19, after step 2 (teach it to double)._

| Area | Metric | Value | Reproduce with (from `core/`) |
|------|--------|-------|-------------------------------|
| **Headline** | Learned agent EV per round | **−0.0098 ± 0.0049** | `./build/train --config ../config/tuned.cfg` |
| | Basic strategy EV per round (same run) | −0.0052 ± 0.0050 | same run |
| | Random play EV per round (same run) | −0.4443 ± 0.0057 | same run |
| | EV gap recovered | **99.0%** | same run |
| **Confirmed** | Agent vs book, 10 fresh seeds × 1M rounds | −0.0071 ± 0.0008 vs −0.0054 ± 0.0006 | `experiments/001-double-down/run.sh confirm` |
| | Shortfall to basic strategy | **0.0017 per round** (was 0.026) | same |
| **Spread** | EV gap recovered, seeds 1–5 | 99.7% ± 1.0 (range 98.8–100.6) | `./build/experiment --config ../config/tuned.cfg --seeds 5` |
| | Agent EV, seeds 1–5 | −0.0076 ± 0.0024 (range −0.0098 to −0.0049) | same |
| **Policy** | Accuracy: overall / two-card / multi-card / pairs | 94.9% / 92.9% / 97.8% / 92.0% | seed 1 |
| | Book doubles (hard 9–11) the agent makes | 18 of 21 (was 3) | seed 1 strategy chart |
| **Engine** | Basic strategy EV, 20M rounds | −0.0061 ± 0.0004 (published house edge for these rules: ≈ −0.005 to −0.006) | `./build/experiment --config ../config/tuned.cfg --seeds 4 --episodes 1000 --eval-rounds 5000000` |
| **Engineering** | Unit tests | 150 passing, 8 suites | `./build/run_tests` |
| | Training throughput | 1.1M episodes/s in the published run (machine shared with other jobs; 2.0–2.7M/s measured idle) | training report |
| | Game simulation | 5.6–6.1M rounds/s (step 1 measurement) | `./build/benchmark` |
| | Greedy decision latency | 3.9–4.5 ns (step 1 measurement) | `./build/benchmark` |
| **Lab** | Experiments recorded | 2 ([`000`](../experiments/000-tuned-baseline/), [`001`](../experiments/001-double-down/)) | [`experiments/`](../experiments/) |
| | Arms run in 001 | 18 explored, 7 confirmed | [`experiments/001-double-down/`](../experiments/001-double-down/) |
| **Counting** | Learned counter's edge per unit bet | not started (step 4) | |
| **Product** | Browser demo | not started (step 3) | |

`±` is a 95% confidence interval. For a single run it comes from the per-round standard error over
200,000 rounds. Across seeds it is a t-interval on the mean of the per-seed values.

## Roadmap status

- [x] **Step 1: lab bench.** Seeds, experiment runner, early stopping on accuracy or EV, payout fix.
- [x] **Step 2: teach it to double.** Visit-scaled α wins; the shortfall to the book is 0.0017/round.
- [ ] **Step 3: browser demo.** WebAssembly build, Play, Coach and Watch modes.
- [ ] **Step 4: card counting.** Hi-Lo true count in the state, learned bet sizing.

## Milestones

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
- **Win rate** isn't tracked as progress. It sits near 43% for almost any sensible policy.
