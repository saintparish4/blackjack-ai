# Progress and metrics

One page for checking where the project stands. Each step appends a milestone below and updates the
scoreboard. Every number here is reproducible from a config file plus a seed, and is measured
through the same `Evaluator`, game engine and rules as the in-engine baselines.

## Scoreboard

_Last updated 2026-09-19, after step 1 (lab bench)._

| Area | Metric | Value | Reproduce with (from `core/`) |
|------|--------|-------|-------------------------------|
| **Headline** | Learned agent EV per round | **−0.0294 ± 0.0045** | `./build/train --config ../config/tuned.cfg` |
| | Basic strategy EV per round (same run) | −0.0023 ± 0.0049 | same run |
| | Random play EV per round (same run) | −0.4458 ± 0.0057 | same run |
| | EV gap recovered | **93.9%** | same run |
| **Spread** | EV gap recovered, 5 seeds | 93.3% ± 1.9 (range 91.2–95.2) | `./build/experiment --config ../config/tuned.cfg --seeds 5` |
| | Agent EV, 5 seeds | −0.0330 ± 0.0079 (range −0.0427 to −0.0268) | same |
| **Policy** | Strategy accuracy: overall / two-card / multi-card | 87.3% / 84.6% / 89.6% | seed 1 |
| | Book doubles (hard 9–11) the agent makes | 3 of 22 | seed 1 strategy chart |
| **Engineering** | Unit tests | 131 passing, 7 suites | `./build/run_tests` |
| | Training throughput | ~2.7M episodes/s | training report |
| | Game simulation | 5.6–6.1M rounds/s | `./build/benchmark` |
| | Greedy decision latency | 3.9–4.5 ns | `./build/benchmark` |
| **Lab** | Experiments recorded | 1 ([`000-tuned-baseline`](../experiments/000-tuned-baseline/)) | [`experiments/`](../experiments/) |
| **Counting** | Learned counter's edge per unit bet | not started (step 4) | |
| **Product** | Browser demo | not started (step 3) | |

`±` is a 95% confidence interval. For a single run it comes from the per-round standard error over
200,000 rounds. Across seeds it is a t-interval on the mean of the per-seed values.

## Roadmap status

- [x] **Step 1: lab bench.** Seeds, experiment runner, early stopping on accuracy or EV, payout fix.
- [ ] **Step 2: teach it to double.** Visit-scaled α, Expected SARSA, Double Q-learning.
- [ ] **Step 3: browser demo.** WebAssembly build, Play, Coach and Watch modes.
- [ ] **Step 4: card counting.** Hi-Lo true count in the state, learned bet sizing.

## Milestones

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
  It's scored separately for two-card hands (where doubling lives) and multi-card hands. States
  where some legal action was never trained are reported as coverage gaps, not scored.
- **Win rate** isn't tracked as progress. It sits near 43% for almost any sensible policy.
