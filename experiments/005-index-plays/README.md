# 005: index plays

**Question:** experiment 002's learned counter, the true count in its playing state, never
learned the index plays. Can any learner rediscover the Illustrious 18 within one true count?
How many episodes does it take, and did the rules predict it?

**Answer:** yes. A learner that is linear in the count finds **16.2 ± 0.7 of 17** on fresh seeds.

- **Pooling across counts is what works.** Q(s, a) = w₀ + w₁·TC, two parameters shared by all
  ten count buckets, rediscovers 14.8 of 17 by 10⁸ episodes and 16.2 by 3 × 10⁸ (seeds
  101–110). It nails several indices exactly on every seed: 16 v T at 0, 12 v 2 at +3,
  13 v 2 at −1.
- **Experiment 002's learner wasn't wrong, just starved.** The same per-bucket table gets 12.3
  of 17 at 3 × 10⁸ episodes, six times 002's budget. The effects of removal predicted 4 × 10⁸
  to 3 × 10¹⁰ episodes per play for it, and that's why.
- **Warm-starting hurts, as it should.** A count-blind prior with 100 pseudo-samples (which is
  exactly deviation learning with Q₀ frozen; a test proves the identity) holds every bucket at
  basic strategy until its own data outweighs the prior: 10.3 of 17.
- **The predicted winner lost.** The analysis below favoured exploring starts. They randomise
  only a round's first decision, so the continuation values every opening comparison leans on
  are never explored. With exploring starts, the linear learner stalls at 9.2, and Monte Carlo
  and Q-learning both sit at the count-blind floor of 2.

## What the rules predict (before any learner runs)

`indices --theory` computes the effects of removal (EoRs) exactly, two solves per card rank,
with the book playing on. Full output: [`theory.txt`](theory.txt).

- **Betting correlation of Hi-Lo: 0.967** (published: 0.97).
- **The single-deck EoRs agree with Griffin's** to within 0.06% per card
  (`CountAnalysisTest.SingleDeckEffectsOfRemovalAreGriffins`).
- **Every linear index estimate lands within one of the book's index,** from the rules alone.
  Each one is −gain/(D·β) for the play's commonest two-card hand.

| Play | Repo (Schlesinger S17) | BJA3 (Farmer's file) | Cacarulo (Farmer's file) | EoR linear estimate | Playing correlation | Learned, linear, seeds 101–110 (mode) |
|---|---|---|---|---|---|---|
| 16 v T | 0 | 0 | 0 | +0.80 | 0.56 | 0 (10/10) |
| 15 v T | +4 | +4 | +4 | +3.63 | 0.79 | +3 (8/10) |
| T,T v 5 | +5 | +5 | +5 | +4.83 | 0.96 | +6 (8/10) |
| T,T v 6 | +4 | +5 | +4 | +4.43 | 0.92 | +5 (8/10) |
| 10 v T | +4 | +4 | n/a | +3.12 | 0.84 | +3 / +4 (4 each) |
| 12 v 3 | +2 | +2 | +2 | +1.66 | 0.71 | +1 (6/10) |
| 12 v 2 | +3 | +4 | +3 | +3.46 | 0.64 | +3 (10/10) |
| 11 v A | +1 | +1 | +1 | +1.08 | 0.83 | +1 (9/10) |
| 9 v 2 | +1 | +1 | +1 | +0.39 | 0.82 | +1 (8/10) |
| 10 v A | +4 | +4 | +3 | +3.41 | 0.94 | +3 (5/10) |
| 9 v 7 | +3 | +4 | +3 | +3.04 | 0.86 | +2 / +3 (5 each) |
| 16 v 9 | +5 | +5 | +4 | +4.56 | 0.54 | +4 (8/10) |
| 13 v 2 | −1 | −1 | −1 | −0.76 | 0.66 | −1 (10/10) |
| 12 v 4 | 0 | 0 | 0 | +0.05 | 0.77 | 0 (10/10) |
| 12 v 5 | −2 | −1 | n/a | −1.57 | 0.82 | −2 (9/10) |
| 12 v 6 | −1 | −1 | n/a | −1.05 | 0.66 | −1 (9/10) |
| 13 v 3 | −2 | −2 | n/a | −2.19 | 0.69 | −2 (10/10) |

The published sets disagree with each other by one on six plays, so ±1 is the honest tolerance.

The learner's misses go the way the EoRs point. It learned +3 for 15 v T and 10 v A, where the
linear estimates are +3.6 and +3.4, and split evenly between +3 and +4 on 10 v T (estimate
+3.1). It learned +4 for 16 v 9, where Cacarulo says +4 and the estimate is +4.6.

**Predicted learnability.** The theory counts the samples of the rarer play needed to read the
gain's sign 1.5 counts either side of the index, then divides by how often the agent gets one:
reach of the opening hand × share of rounds at that count × exploration share. That predicted,
per play:

| Learner | Predicted episodes per play |
|---|---|
| ε-greedy table (002's) | 4 × 10⁸ – 3 × 10¹⁰ |
| Exploring starts, table | 2 × 10⁷ – 1.5 × 10⁹ |
| ε-greedy, linear | 1 × 10⁸ – 1.5 × 10⁹ |
| Exploring starts, linear | 6 × 10⁶ – 8 × 10⁷ |

## Exploration (seeds 1–5): plays rediscovered of 17

Mean ± t-based 95% CI over five seeds. The count-blind floor is 2.

| Arm | 10⁶ | 3×10⁶ | 10⁷ | 3×10⁷ | 10⁸ | 3×10⁸ |
|---|---|---|---|---|---|---|
| **linear** (ε) | 7.2 ± 3.2 | 9.6 ± 1.7 | 11.2 ± 1.0 | 13.8 ± 1.6 | 15.4 ± 1.1 | **16.6 ± 0.7** |
| ucb | 4.0 ± 1.5 | 6.4 ± 1.4 | 9.0 ± 1.8 | 11.2 ± 1.8 | 13.0 ± 0.9 | 12.2 ± 1.6 |
| table (002's learner) | 5.0 ± 0.9 | 6.8 ± 1.8 | 7.4 ± 0.7 | 7.0 ± 2.3 | 9.8 ± 1.6 | 12.4 ± 1.9 |
| frozen-prior (warm start) | 3.2 ± 1.0 | 4.2 ± 1.4 | 5.8 ± 1.0 | 6.6 ± 1.1 | 9.0 ± 1.8 | 10.8 ± 2.0 |
| pooled-prior | 3.4 ± 1.1 | 3.6 ± 1.4 | 4.6 ± 1.4 | 6.2 ± 2.4 | 8.0 ± 3.2 | 10.6 ± 2.9 |
| linear-es (predicted best) | 6.4 ± 2.9 | 6.6 ± 1.7 | 7.0 ± 1.5 | 7.4 ± 1.4 | 8.6 ± 1.9 | 9.2 ± 2.2 |
| td-es | 3.4 ± 1.4 | 3.0 ± 1.2 | 3.0 ± 1.8 | 2.6 ± 1.1 | 2.8 ± 1.0 | 2.8 ± 1.0 |
| mc-es | 4.6 ± 3.4 | 3.0 ± 1.5 | 3.2 ± 0.6 | 2.6 ± 0.7 | 2.4 ± 0.7 | 2.4 ± 0.7 |

## Confirmation (fresh seeds 101–110, pre-registered in [`candidates.txt`](candidates.txt))

| Arm | 10⁶ | 3×10⁶ | 10⁷ | 3×10⁷ | 10⁸ | 3×10⁸ |
|---|---|---|---|---|---|---|
| **linear** | 8.5 ± 1.6 | 10.0 ± 1.0 | 11.8 ± 1.3 | 12.6 ± 0.9 | 14.8 ± 0.7 | **16.2 ± 0.7** |
| ucb | 4.5 ± 0.9 | 5.8 ± 1.0 | 9.0 ± 1.5 | 10.6 ± 1.5 | 12.2 ± 0.9 | 13.7 ± 1.1 |
| table | 3.8 ± 1.6 | 4.6 ± 1.3 | 5.9 ± 1.1 | 7.6 ± 1.6 | 8.8 ± 1.8 | 12.3 ± 1.6 |
| frozen-prior | 3.6 ± 0.9 | 4.1 ± 1.1 | 4.6 ± 1.2 | 5.8 ± 1.3 | 8.0 ± 0.9 | 10.3 ± 0.7 |

**Episodes to rediscovery, per play.** The first checkpoint at which at least 8 of the 10
seeds have the play, and at least 8 still do at every later checkpoint (not necessarily the
same 8):

| Play | linear | ucb | table | frozen-prior |
|---|---|---|---|---|
| 16 v T | 10⁷ | 3×10⁷ | 3×10⁸ | 10⁸ |
| 15 v T | 10⁷ | 10⁸ | 3×10⁸ | > 3×10⁸ |
| T,T v 5 | 10⁸ | 10⁸ | 3×10⁸ | > 3×10⁸ |
| T,T v 6 | 3×10⁸ | > 3×10⁸ | > 3×10⁸ | > 3×10⁸ |
| 10 v T | 3×10⁸ | > 3×10⁸ | > 3×10⁸ | > 3×10⁸ |
| 12 v 3 | 3×10⁷ | 3×10⁷ | 10⁸ | 3×10⁸ |
| 12 v 2 | 10⁷ | 10⁸ | > 3×10⁸ | > 3×10⁸ |
| 11 v A | 3×10⁷ | > 3×10⁸ | > 3×10⁸ | 3×10⁸ |
| 9 v 2 | 10⁸ | > 3×10⁸ | 3×10⁸ | > 3×10⁸ |
| 10 v A | > 3×10⁸ | > 3×10⁸ | > 3×10⁸ | > 3×10⁸ |
| 9 v 7 | 10⁸ | > 3×10⁸ | > 3×10⁸ | > 3×10⁸ |
| 16 v 9 | 3×10⁸ | 3×10⁸ | > 3×10⁸ | > 3×10⁸ |
| 13 v 2 | 3×10⁶ | 10⁷ | 10⁷ | 10⁷ |
| 12 v 4 | 10⁶ | 10⁷ | 3×10⁸ | 10⁸ |
| 12 v 5 | 10⁶ | 10⁷ | 3×10⁸ | 3×10⁸ |
| 12 v 6 | 10⁶ | 10⁷ | 10⁷ | 3×10⁶ |
| 13 v 3 | 3×10⁶ | 3×10⁶ | 3×10⁷ | 3×10⁶ |

The low-count stiff hands come first: 12 v 4–6 and 13 v 2–3, whose indices sit where most rounds
are dealt. The positive-count doubles come last, at counts reached in a few percent of rounds.
10 v A isn't held by 8 seeds at any checkpoint, though 7 of 10 are within one at 3 × 10⁸.

## What the prediction got right and wrong

- **Right: the ranking among the ε learners.** Linear beats the table by orders of magnitude in
  episodes, and the table needs roughly 10⁸–10⁹ for most plays, as predicted. 002's 5 × 10⁷ was
  short by about the predicted factor.
- **Wrong: conservative by about 10× for the ε learners.** The 2-sigma requirement on both sides
  of the index, and SD 2 for every double, overstate the samples needed.
- **Wrong: exploring starts.** The sample-count model scored a decision by how often it gets
  tried and ignored the quality of the values its targets lean on. With exploring starts, every
  decision after the first is greedy, so HIT's continuation is never explored. The linear
  learner with exploring starts stalls at 9.2, and the tabular ones never leave the floor.

## Reproduce

```bash
experiments/005-index-plays/run.sh theory    # ~20 s
experiments/005-index-plays/run.sh explore   # 8 arms x 5 seeds; ~30 min on 8 cores (not timed end to end)
experiments/005-index-plays/run.sh confirm   # 4 arms x 10 seeds; 26 min on 8 cores
```

Per seed, checkpoint and play: `explore/<arm>/indices.csv`, `confirm/<arm>/indices.csv`.
