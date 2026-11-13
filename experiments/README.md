# Lab notebook

Each folder is one experiment: the configs it ran, the per-seed `results.csv`, the across-seed
`summary.csv`, and a write-up. Negative results are written up too.

The ground rules:

- **Same bench for everyone.** Every arm is trained and scored by `./build/experiment` on the same
  seeds, rules and evaluator, next to basic strategy and random play measured the same way.
- **EV is the verdict.** Cell-match accuracy is reported but secondary. It weights a hard 4 the same
  as a hard 16, and only one of those comes up constantly.
- **Reproducible from the folder.** A config plus a seed range regenerates every number. Per-seed
  models and logs (`seed-*/`) are gitignored because they regenerate on demand.

## Index

| # | Experiment | Question | Result |
|---|------------|----------|--------|
| 000 | [tuned-baseline](000-tuned-baseline/) | How much does `tuned.cfg` vary from seed to seed? | Gap recovered 93.3% ± 1.9 over 5 seeds. Recorded before 001's engine and yardstick fixes, so not comparable with later numbers |
| 001 | [double-down](001-double-down/) | Which learner change teaches the agent to double, and does it close the EV gap to the book? | Visit-scaled α, 1/(1+n): EV −0.0071 ± 0.0008 vs book −0.0054 ± 0.0006 on 10 fresh seeds. The shortfall fell from 0.026 to 0.0017 per round. Also fixed two engine rules, the book chart and split credit |
| 002 | [card-counting](002-card-counting/) | Can the agent learn to beat the house by counting cards? | It learns *when to bet*: +0.27% ± 0.05 per unit wagered on 10 fresh seeds (flat basic strategy −0.58%), beating textbook Hi-Lo's ramp and tying Hi-Lo + Illustrious 18. Putting the count into its playing state failed: −0.10% ± 0.13 |
| 003 | [ground-truth](003-ground-truth/) | Does anything survive contact with code nobody here wrote? | An exact solver agrees with Farmer's CA and Wizard of Odds to every published digit on five rule sets; the engine matches it on 104 audited quantities (max \|z\| 2.59); three rule flags and one preset were fiction and are deleted |

## Algorithm leaderboard

From experiment 001's confirmation phase: seeds 101–110, 5M training episodes, Vegas Strip rules,
1M evaluation rounds per policy per seed. `±` is a t-based 95% CI across seeds.

| Agent | EV per round | EV gap recovered | Two-card accuracy | Pair accuracy |
|-------|--------------|------------------|-------------------|---------------|
| Basic strategy (reference) | −0.0054 ± 0.0006 | 100% by definition | 100% | 100% |
| **Q-learning, visit α (ω = 1.0)**, now `tuned.cfg` | **−0.0071 ± 0.0008** | **99.6% ± 0.3** | 94.1% | 93.2% |
| Q-learning, visit α (ω = 0.8) | −0.0082 ± 0.0007 | 99.4% ± 0.2 | 91.0% | 90.5% |
| Double Q-learning, constant α | −0.0232 ± 0.0011 | 95.9% ± 0.3 | 81.5% | 78.9% |
| Q-learning, constant α = 0.05 (old `tuned.cfg`) | −0.0313 ± 0.0011 | 94.1% ± 0.2 | 78.1% | 72.0% |
| Expected SARSA, constant α | −0.0336 ± 0.0032 | 93.6% ± 0.7 | 78.0% | 74.2% |
| Random legal play (reference) | about −0.444 | 0% by definition | n/a | n/a |

## Card-counting leaderboard

From experiment 002's confirmation phase: seeds 101–110, Vegas Strip, 75% penetration, 1–8 unit
bets, 10M rounds per policy per seed, all policies on the same shoes.

| Policy | EV per unit wagered | Units won / 100 rounds | Desirability index |
|---|---|---|---|
| **Basic strategy + learned bets** | **+0.27% ± 0.05** | **+0.74 ± 0.14** | **1.62 ± 0.31** |
| Hi-Lo + Illustrious 18 (textbook) | +0.22% ± 0.03 | +0.32 ± 0.04 | 1.41 ± 0.18 |
| Learned play + learned bets | +0.16% ± 0.07 | +0.40 ± 0.19 | 0.96 ± 0.42 |
| Hi-Lo ramp (textbook) | +0.11% ± 0.04 | +0.16 ± 0.07 | 0.71 ± 0.29 |
| Learned counter (count in state, 50M episodes) | −0.10% ± 0.13 | −0.18 ± 0.22 | −0.55 ± 0.74 |
| Flat basic strategy | −0.58% ± 0.02 | −0.58 | n/a |

Rank any folder of arms with `python analysis/leaderboard.py <folder> [--markdown]`.

## Running one

From `core/`:

```bash
./build/experiment --config ../experiments/<folder>/<arm>.cfg --seeds 5 \
    --out ../experiments/<folder> --name <arm>
```

Arm configs extend a frozen base (a copy of the config the experiment started from) and change only
what the arm is testing, so later edits to `config/tuned.cfg` can't move old results. Experiments
with several arms ship a `run.sh` that reruns the whole thing.
