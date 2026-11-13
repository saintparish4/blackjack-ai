# 004: regret, not cell matches

**Question:** experiment 001 left the agent 0.0017 per round short of basic strategy. Where
exactly does that EV go? How much is the agent's state representation unable to hold, and how
much did it simply not learn?

**Answer:** the shortfall was smaller than published, and almost all of it is learning.

- **The confirmed shortfall is 0.112% ± 0.029 per round, not 0.17%.** Scored exactly, the ten
  agents give up 0.00112 ± 0.00029 per round to the book. Experiment 001's figure was a
  difference of two Monte Carlo estimates, and the book's (−0.0054 ± 0.0006) came out about
  0.0004 lucky against its exact −0.00582.
- **The representation costs 0.0015% per round.** The best policy the agent's own `State` can
  express (TD*) sits 0.0015% below composition-dependent optimal play. Of the agents' shortfall
  to optimal, 99% is learning error.
- **The book chart isn't the best total-dependent policy.** TD* differs from it in one cell: it
  stands on multi-card hard 16 vs T (the agent's `State` tells two-card from multi-card 16),
  which is worth 0.0016% per round.
- **No single leak dominates.** Missed doubles (hard 9–11, soft hands, 5,5) and standing on
  two-card 16 vs T make up the first half of the bill. Six of the ten agents stand on 16 vs T.
- **An independent reproduction.** Farmer's code, playing each learned policy exported from its
  Q-table, gives the same exact EV to nine decimals, for all ten agents.

## Method

The solver computes the exact EV of any policy ([experiment 003](../003-ground-truth/)). Two references:

- **CD optimal:** composition-dependent play, −0.578793657%.
- **TD\*:** the best total-dependent policy over the agent's states, found by policy iteration
  from the book. Each `State` gets the action with the highest reach-weighted exact value under
  the current policy; re-solve; repeat. It converged in one step, to −0.580301250%. States share
  actions across compositions, so this is a local optimum, and its EV a lower bound on the
  representation's ceiling.

The bill is the performance-difference identity:

    EV(TD*) - EV(agent) = sum over decisions n the agent reaches of
                          reach(n) x [Q_TD*(n, TD*'s play) - Q_TD*(n, agent's play)]

Grouping the sum by the agent's `State` itemises the learning error exactly. The report checks
that the lines sum to the total: they did, to 10⁻¹⁷.

## Confirmed agents (experiment 001's visit-α arm, seeds 101–110, 5M episodes)

| | Exact, per round (t-based 95% CI over 10 agents) |
|---|---|
| Agent EV | −0.693986% ± 0.029332 (range −0.794% to −0.649%) |
| Shortfall to the book | **0.112133% ± 0.029332** |
| Shortfall to CD optimal | 0.115192% ± 0.029332 |
| … representation ceiling (CD − TD*) | 0.001508% |
| … learning error (TD* − agent) | 0.113684% ± 0.029332 |

Every EV is exact, so the interval is training variation alone. It's wide because one seed (109)
lost twice as much as the median. The constant-α baseline from 001, scored the same way, gives
up 2.4825% ± 0.0743 to the book. So experiment 001's step-size fix cut the shortfall from 0.0248
to 0.0011 per round. That's 95.5%, not the 94% experiment 001 reported, and the endpoint was
overstated.

## The bill (mean over the ten agents, largest first)

| State | Loss / round | Share | TD* plays | Agents that don't |
|---|---|---|---|---|
| hard 11 vs T (2 cards) | 0.0080% | 7.0% | double | 1 (hit) |
| hard 16 vs T (2 cards) | 0.0053% | 4.7% | hit | 6 (stand) |
| hard 9 vs 4 (2 cards) | 0.0050% | 4.4% | double | 3 (hit) |
| hard 9 vs 3 (2 cards) | 0.0049% | 4.3% | double | 7 (hit) |
| hard 10 vs 9 (2 cards) | 0.0042% | 3.7% | double | 5 (hit) |
| 5,5 vs 2 | 0.0041% | 3.6% | double | 5 (hit) |
| soft 18 vs 4 (2 cards) | 0.0038% | 3.3% | double | 6 (stand) |
| 5,5 vs 3 | 0.0037% | 3.3% | double | 4 (hit) |
| soft 16 vs 5 (2 cards) | 0.0034% | 3.0% | double | 7 (hit) |
| soft 16 vs 6 (2 cards) | 0.0031% | 2.7% | double | 4 (hit) |

Twenty states make up 62% of the loss. Full output: [`regret.txt`](regret.txt).

- **One agent's 11 vs T.** Hitting where it should double costs that agent 0.08% per round on
  its own: the largest single line, from a single seed.
- **Doubles still dominate.** Experiment 001 fixed doubling on average, but doubling is still
  where most of the residual lives, because DOUBLE is the action with the noisiest returns and
  the fewest visits.
- **16 vs T.** It's marginal in EV but common: 1.8% of rounds reach it.

## Independent reproduction

`regret --export-farmer` writes each agent's policy as one line per state.
`tools/farmer/policy_ev` loads those lines into Farmer's `IndexStrategy` with no count variation,
and his `BJPlayer` plays it:

| Policy | Solver | Farmer's code |
|---|---|---|
| Book chart | −0.581852995% | −0.581852995% |
| TD* | −0.580301250% | −0.580301250% |
| Book chart, Downtown (2D H17, surrender) | −0.504177180% | −0.504177180% |
| Confirmed agents, seeds 101–110 | all ten | identical to nine decimals |

Nine-decimal agreement across ten learned policies, from a second implementation, settles the
question of whether the learned policies are graded correctly.

## Reproduce

```bash
cd core
./build/experiment --config ../experiments/001-double-down/arms/visit-omega-1.0.cfg --seeds 10 \
    --seed-base 101 --eval-rounds 10000 --out /tmp/004 --name confirm
./build/regret --models "$(for s in $(seq 101 110); do printf "/tmp/004/confirm/seed-$s/agent,"; done)" \
    --top 20 --export-farmer /tmp/004/agent | tee ../experiments/004-regret/regret.txt
for i in $(seq 0 9); do ../tools/farmer/bin/policy_ev 6 n n n /tmp/004/agent-$i.policy; done
```

Training doesn't depend on evaluation length, so `--eval-rounds 10000` regenerates exactly the
agents experiment 001 confirmed (its pair accuracy, 93.20% ± 2.18, reproduces to the digit).
