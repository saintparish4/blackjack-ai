# 001: teach it to double

**Question:** the Q-learning agent almost never doubled. It hit hard 10 and 11 where basic strategy
doubles, and that was its biggest EV leak. Which change to the learner fixes that, and does it close
the EV gap to basic strategy?

**Answer:** a visit-count-scaled step size. With α = 1/(1+n), each Q-value becomes the running mean
of its targets. It cut the agent's EV shortfall to basic strategy from **0.026 to 0.0017 per round**
(a 94% cut) on ten fresh seeds. The agent now makes 18 of the 21 hard doubles the book makes,
where it used to make 3. Expected SARSA, Double Q-learning and an undiscounted return did little or
nothing on their own. A residual 0.0017 per round is still statistically real, and 4× more training
doesn't move it.

## Setup

- **Rules:** Vegas Strip (6 decks, dealer stands on soft 17, no surrender, one split, no double after
  split), 5M training episodes per seed.
- **Frozen base:** every arm extends [`base.cfg`](base.cfg), a frozen copy of `tuned.cfg` as it stood
  (Q-learning, α = 0.05, γ = 0.95, ε 1 → 0.05). Arm configs are in [`arms/`](arms/) and change only
  what the arm tests.
- **Two phases**, so that picking the best arm doesn't flatter it:
  - **Explore:** every arm on seeds 1–5, 200k evaluation rounds per policy per seed.
  - **Confirm:** the planned arms plus the two best explorers, pre-registered in
    [`candidates.txt`](candidates.txt) before the run, on fresh seeds 101–110 with 1M evaluation
    rounds per policy per seed.
- **Reproduce everything:** `experiments/001-double-down/run.sh all` (about 4 minutes on 8 cores).
  Rerunning it gives byte-identical `results.csv` files.

## Confirmation results (seeds 101–110, 1M rounds per policy per seed)

`±` is a t-based 95% CI across the ten seeds.

| Arm | Agent EV / round | Book EV / round | Gap recovered | Two-card acc. | Pair acc. | Overall acc. |
|---|---|---|---|---|---|---|
| **visit α, ω = 1.0** | **−0.0071 ± 0.0008** | −0.0054 ± 0.0006 | **99.6% ± 0.3** | 94.1% | 93.2% | 95.9% |
| visit α, ω = 0.8, ε floor 0.10 | −0.0079 ± 0.0007 | −0.0054 ± 0.0006 | 99.4% ± 0.2 | 91.2% | 92.1% | 94.5% |
| visit α, ω = 0.8 (planned arm) | −0.0082 ± 0.0007 | −0.0054 ± 0.0006 | 99.4% ± 0.2 | 91.0% | 90.5% | 94.0% |
| Double Q-learning | −0.0232 ± 0.0011 | −0.0054 ± 0.0006 | 95.9% ± 0.3 | 81.5% | 78.9% | 86.4% |
| Baseline (constant α = 0.05) | −0.0313 ± 0.0011 | −0.0054 ± 0.0006 | 94.1% ± 0.2 | 78.1% | 72.0% | 83.6% |
| Expected SARSA | −0.0336 ± 0.0032 | −0.0054 ± 0.0006 | 93.6% ± 0.7 | 78.0% | 74.2% | 83.8% |
| γ = 1 (undiscounted) | −0.0355 ± 0.0028 | −0.0054 ± 0.0006 | 93.1% ± 0.6 | 77.1% | 76.3% | 83.5% |

Per-seed numbers: `confirm/<arm>/results.csv`. On 2 of the 10 seeds (101 and 104) the ω = 1.0
agent measured at or above the book. Averaged over all ten, it sits 0.0017 per round below. The
intervals just miss overlapping, so at this precision the residual is real. It is 0.17% of a bet.

The exploration phase (18 arms, seeds 1–5) is in [`explore/`](explore/). The full table:
`python analysis/leaderboard.py experiments/001-double-down/explore`.

## What each arm taught us

- **Visit-scaled α is the fix.** DOUBLE pays ±2, so it has the noisiest returns, and ε-greedy tries
  it rarely. At a constant α = 0.05, each Q-value tracks roughly its last 20 targets. In the
  baseline, doubling 11 scored +0.48 against a 4 and −0.30 against a 5 on the same run. With
  α = 1/(1+n) the estimate averages every sample it has seen, and the doubling column fills in.
  ω between 0.8 and 1.0 all work. ω = 0.6 forgets too fast (−0.0127).
- **Double Q-learning helps a little on its own** (−0.031 → −0.023). The maximisation bias it removes
  is real, but it is the smaller problem. With visit α on top it adds nothing measurable.
- **Expected SARSA doesn't help.** Averaging over the next action lowers the variance of the
  bootstrap target. But DOUBLE's target has no next action: it's the terminal ±2 payout, and that's
  where the noise is.
- **γ = 1 doesn't matter here.** The objective is undiscounted EV, so 0.95 is technically a bias. In
  practice it's swamped by estimation noise, and it stays within noise even with visit α.
- **More training moves accuracy, not EV.** 20M episodes lifts pair accuracy to 96% but leaves EV at
  −0.0077. The last 0.0017 per round sits in marginal cells: soft doubles (A,2–A,5), 9 vs 3, 10 vs
  9, 16 vs T. There the top two actions are within a few hundredths of each other, below what the
  Q-estimates can resolve.

## Three bugs found along the way

Three problems had nothing to do with the learner. Each is its own commit, and every result above
comes after all three. The first two turned up while checking the yardstick before any arm ran. The
third turned up because the first pass put visit α at −0.0115 against a book of −0.0062, and that
gap needed explaining.

1. **The engine favoured the player.** A+T after a split paid 3:2 as a natural, and split aces could
   keep drawing. Basic strategy measured −0.002 per round, when the published house edge for these
   rules is about −0.005 to −0.006. After the fix the book measures −0.0061 ± 0.0004 over 20M
   rounds.
2. **The reference chart was simplified.** It had no soft doubles, and its pair splits assumed
   double-after-split, which the engine doesn't allow. The yardstick was weaker than real basic
   strategy, and it marked correct soft doubles as mistakes.
3. **Split credit was broken, and the audit couldn't see it.** SPLIT was valued as one hand instead
   of two, so the agent split 2s, 3s, 6s and 7s against strong upcards: 27 of 100 pair cells wrong.
   The audit only scored hands that can't split, so none of this showed up in its numbers.
   - A Monte Carlo total for SPLIT overcorrected into under-splitting, because it values the split
     under the exploring play that follows it.
   - A TD target over both hands' states, γ·(V₁ + V₂), got it to 4 of 100 wrong, all close calls.
   - The audit now scores pairs, and the chart shows them.

## Outcome

- `config/tuned.cfg` and `config/default.cfg` now use `alpha_schedule = visit`, `alpha_omega = 1.0`.
  The published run (seed 1) measures −0.0098 ± 0.0049 against the book's −0.0052 ± 0.0050, and the
  two intervals overlap. Seeds 1–5 of the new `tuned.cfg` reproduce the `visit-omega-1.0` exploration
  arm byte for byte.
- **Next for the residual:** an exact dynamic-programming solver would give true per-action EVs.
  That would show which cells still cost the most, and let the audit weight mistakes by their cost
  instead of counting cells.
