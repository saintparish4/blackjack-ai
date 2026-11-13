# 003: ground truth

**Question:** every number this project published was checked against something inside the
project: a hand-typed chart, an evaluator, an engine, all written here. Does any of it survive
contact with code nobody here wrote?

**Answer:** yes, and now it can be checked on every push.

- **An exact solver.** It agrees with Eric Farmer's combinatorial analyzer on every digit Farmer
  prints, and with the Wizard of Odds house-edge table on every digit it publishes, on five rule
  sets. That covers 540 two-card strategy decisions per rule set (54 hands × 10 upcards), every upcard's EV and the
  dealer's outcome table.
- **The engine plays the rules it says it plays.** Over 20M rounds per pass on each of four
  presets, about 100 audited quantities (dealer bust rate by upcard, naturals, EV split by the
  opening play) land within 2.6 standard errors of their exact values.
- **The repo's book chart is the textbook chart.** It's worth exactly −0.581853% per round,
  0.0031% behind composition-dependent play, the gap Wizard of Odds quotes.
- **Three rule flags were fiction.** `doubleAfterSplit`, `maxSplits` and `resplitAces` were set
  by every preset and read by nothing, and the `european` preset was `vegas-strip` under another
  name. All four are deleted.

## The references

| Reference | What it is | How it's used |
|---|---|---|
| Farmer's CA, `github.com/possibly-wrong/blackjack` at `a1f7dbb` | An exact combinatorial analyzer (GPL-3.0) | `tools/farmer/run_farmer.sh` fetches and builds it; `compare_solver.sh` diffs its output with `./build/solve`; CI runs that diff on every push |
| Wizard of Odds house-edge calculator | Its page embeds a 3,456-entry table of exact edges (composition-dependent, reshuffled every hand) | Read off with a 10-line script (Reproduce, below) |

## Solver against references (exact, off the top of a full shoe)

| Rules | Solver | Farmer | Wizard of Odds | Strategy table, EV by upcard, dealer table |
|---|---|---|---|---|
| 6D S17 NDAS (the engine's Vegas Strip) | −0.578793657% | −0.578793657% | −0.578794% | identical, line for line |
| 6D H17 DAS | −0.668950590% | −0.668950590% | −0.668951% | identical |
| 1D H17 NDAS | −0.168766712% | −0.168766712% | −0.168767% | identical |
| 2D H17 NDAS LS (Downtown) | −0.485748350% | −0.485748350% | −0.485748% | identical |
| 8D S17 NDAS LS (Atlantic City) | −0.531383781% | −0.531383781% | −0.531384% | identical |

Every rule set is one split, double on any two cards, 3:2, dealer peeks.

The finite solver also converges to an independently written infinite-deck solver at the
textbook rate. EV(D) − EV(∞), times D, settles at 0.6484% between 256 and 4,096 decks. That's
Griffin's 1/D law, and it checks the dealer rules, peek, doubles and splits with code that shares
nothing but a struct.

## What the references said about the repo

- **The book chart.** It's worth −0.581852995% per round off the top: 0.003059% behind optimal
  play. Wizard of Odds puts total-dependent strategy 0.0031% behind in six decks, so the chart in
  `Evaluator.cpp` is the standard one.
- **"In line with the published house edge (about −0.005 to −0.006)".** The README's check was
  right, and can now be exact: −0.5819% off the top. The README's 20M-round figure,
  −0.0061 ± 0.0004, was measured with a cut card at 75%. It's consistent with the exact value
  plus a small cut-card effect (below).
- **Experiment 002's flat basic strategy, −0.58% ± 0.02** over 100M rounds, contains the exact
  value.
- **Split tens.** The engine splits only identical ranks (K,K, not K,Q). Farmer splits any two
  tens. Optimal play never splits tens off the top, so the two give identical EVs, and the solver
  models the engine's rule exactly for any policy that does split them.

## The engine against the solver

`./build/audit --rounds 20000000 --cut-card-rounds 20000000` (seed 1) plays every round from a
freshly shuffled shoe, which is what the solver computes. It runs two passes:

- **Dealer pass:** a stand-on-everything player, so the dealer's draws are isolated.
- **Book pass:** the chart, with EV split by opening play.

| Preset | Lines | Largest \|z\| | Book EV, exact | Engine (± SE) |
|---|---|---|---|---|
| vegas-strip | 25 | 2.06 | −0.005819 | −0.005673 ± 0.000253 |
| downtown | 27 | 1.52 | −0.005042 | −0.004959 ± 0.000250 |
| atlantic-city | 27 | 2.59 | −0.005332 | −0.005032 ± 0.000250 |
| single-deck | 25 | 1.70 | −0.002209 | −0.002018 ± 0.000254 |

Across the 104 lines, the largest |z| is about what 104 independent normal draws produce. At
this sample size there's no engine bug to find. Full output:
[`audit-20m.txt`](audit-20m.txt).

**A new measurement: the cut-card effect.** The same book, dealt to 75% penetration, against the
off-the-top exact EV:

| Preset | Cut-card cost per round (± SE) | Wizard of Odds' cut-card adjustment |
|---|---|---|
| single-deck | 0.108% ± 0.025 | 0.113% |
| downtown (2 decks) | 0.103% ± 0.025 | 0.063% |
| atlantic-city (8 decks) | 0.040% ± 0.025 | 0.014% |
| vegas-strip (6 decks) | 0.026% ± 0.025 | 0.020% |

- **Single deck:** agrees with Wizard of Odds.
- **6 and 8 decks:** the effect is below what 20M rounds can resolve.
- **Two decks:** 1.6 SE above Wizard of Odds' figure. The penetration Wizard of Odds assumes
  isn't stated, so this isn't a disagreement yet.

## Changes

- `GameRules`: deleted `doubleAfterSplit`, `maxSplits`, `resplitAces` and the `european` preset.
  Its comment now states the rules the engine actually plays, and a test fails if two presets
  are ever identical again.
- `tools/farmer/`, `solve`, `audit`, `solver/`, and a CI job that diffs the solver against
  Farmer on every push.
- `TrainerTest` fixtures each get their own temp directories, which let ctest run in parallel.

## Reproduce

```bash
cd core
for r in "6 n n n" "6 y y n" "1 y n n" "2 y n y" "8 n n y"; do ../tools/farmer/compare_solver.sh $r; done
./build/audit --rounds 20000000 --cut-card-rounds 20000000

# Wizard of Odds: the same five rule sets, read off the calculator page's edge table
curl -sL -A "Mozilla/5.0" https://wizardofodds.com/games/blackjack/calculator/ -o /tmp/woo.html
python3 - <<'EOF'
import re
t = {tuple(map(int, k.split(']['))): float(v)
     for k, v in re.findall(r'edgeTable\[([\d\]\[]+)\]=(-?[\d.]+);', open('/tmp/woo.html').read())}
# index: decks(1,2,4,5,6,8), H17, DAS, double rule, resplit-to, RSA, hit split aces, OBO (peek), surrender
for name, d, h, das, ls in [("6D S17 NDAS", 4,0,0,0), ("6D H17 DAS", 4,1,1,0), ("1D H17 NDAS", 0,1,0,0),
                            ("2D H17 NDAS LS", 1,1,0,1), ("8D S17 NDAS LS", 5,0,0,1)]:
    print(f"{name:16s} {-t[(d,h,das,0,0,0,0,1,ls)]:+.6f}%")
EOF
```
