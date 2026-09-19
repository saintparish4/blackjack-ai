# 000: tuned baseline

**Question:** how much do the published numbers depend on the seed?

**Setup:** `config/tuned.cfg` unchanged (Q-learning, α = 0.05, γ = 0.95, 5M episodes, Vegas Strip
rules), seeds 1–5, 200,000 evaluation rounds per policy per seed.

```bash
cd core
./build/experiment --config ../config/tuned.cfg --seeds 5 \
    --out ../experiments --name 000-tuned-baseline
```

## Results

| Seed | Agent EV (95% CI) | Basic strategy EV | Gap recovered | Accuracy | Two-card | Multi-card |
|------|-------------------|-------------------|---------------|----------|----------|------------|
| 1 | −0.0294 ± 0.0045 | −0.0023 | 93.9% | 87.3% | 84.6% | 89.6% |
| 2 | −0.0268 ± 0.0045 | −0.0056 | 95.2% | 87.3% | 82.9% | 91.1% |
| 3 | −0.0357 ± 0.0046 | −0.0018 | 92.4% | 87.9% | 82.5% | 92.5% |
| 4 | −0.0427 ± 0.0047 | −0.0040 | 91.2% | 86.3% | 80.0% | 91.8% |
| 5 | −0.0302 ± 0.0045 | −0.0030 | 93.9% | 87.9% | 83.3% | 91.8% |
| **Mean** | **−0.0330 ± 0.0079** | −0.0033 ± 0.0019 | **93.3% ± 1.9** | 87.3% ± 0.8 | 82.7% ± 2.1 | 91.4% ± 1.3 |

Raw numbers: [`results.csv`](results.csv), [`summary.csv`](summary.csv).

## Reading

- On every seed the agent's EV interval sits clear of basic strategy's. The ~0.03 per round gap is
  real, and it's the target for experiment 001.
- Seed-to-seed spread in agent EV (sd 0.006) is larger than any single run's evaluation error
  (±0.0045 half-width), so one run is not enough to compare two algorithms. That's the reason the
  bench exists.
- Two-card accuracy varies most (80.0–84.6%), and it's where DOUBLE lives. On seed 1 the agent makes
  3 of the 22 hard 9–11 doubles the book makes. Its Q-estimate for doubling 11 is +0.48 against a 4
  and −0.30 against a 5, when the true value is strongly positive against both. That's noise, and
  experiment 001 goes after it.
