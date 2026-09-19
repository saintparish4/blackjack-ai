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
| 000 | [tuned-baseline](000-tuned-baseline/) | How much does `tuned.cfg` vary from seed to seed? | Gap recovered 93.3% ± 1.9 over 5 seeds; EV −0.033 ± 0.008 vs basic strategy −0.003 |

## Algorithm leaderboard

Mean over seeds 1–5 of `tuned.cfg`-length runs (5M episodes), Vegas Strip rules, 200k evaluation
rounds per policy per seed. `±` is a t-based 95% CI across seeds.

| Agent | EV per round | EV gap recovered | Two-card accuracy | Source |
|-------|--------------|------------------|-------------------|--------|
| Basic strategy (reference) | −0.0033 ± 0.0019 | 100% by definition | 100% | 000 |
| Q-learning, `tuned.cfg` | −0.0330 ± 0.0079 | 93.3% ± 1.9 | 82.7% ± 2.1 | 000 |
| Random legal play (reference) | −0.4450 ± 0.0013 | 0% by definition | n/a | 000 |

## Running one

From `core/`:

```bash
./build/experiment --config ../experiments/<folder>/<arm>.cfg --seeds 5 \
    --out ../experiments/<folder> --name <arm>
```

Arm configs use `extends = ../../config/tuned.cfg` and change only what the arm is testing.
