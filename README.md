# Blackjack AI

[![CI](https://github.com/saintparish4/blackjack-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/saintparish4/blackjack-ai/actions/workflows/ci.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/build-CMake-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/tests-114%20passing-3fb950)](core/tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A C++17 Q-learning agent that learns blackjack strategy from scratch through self-play — no rules,
no lookup tables, no supervision. Starting from an all-zero Q-table, 5 million hands of trial and
error recover **92% of the expected-value gap** between random play and textbook basic strategy,
at roughly 2.9M episodes/second and 3.2ns per decision.

Ships with an interactive play mode, a colour-coded strategy chart, an exhaustive post-training
policy audit measured against in-engine baselines, and Python visualisation scripts.

<p align="center">
  <img src="docs/images/strategy_chart.png" alt="Learned strategy chart, colour-coded against basic strategy" width="620">
</p>

<p align="center">
  <em>The learned policy graded cell-by-cell against basic strategy —
  <strong>green</strong> matches, <strong>red</strong> diverges, <strong>yellow</strong> is low-confidence.
  Printed at the end of every training run.</em>
</p>

---

## Results

Trained with [`config/tuned.cfg`](config/tuned.cfg) on Vegas Strip rules (6 decks, dealer stands
soft 17, no surrender). Full output: [`docs/training_report.txt`](docs/training_report.txt).

### Expected value — the metric that matters

Every figure below is measured on the same game engine, rules and evaluator, over 200,000 rounds
each, so the three are directly comparable.

| Policy | EV per round (95% CI) |
|--------|----------------------|
| Basic strategy (reference) | **−0.0034** ± 0.0049 |
| **Learned agent** | **−0.0368** ± 0.0045 |
| Random legal play | −0.4442 ± 0.0057 |

**EV gap recovered: 92.4%** of the distance from random play to book play.

The agent is genuinely good and genuinely not optimal. Its interval does not overlap basic
strategy's, so the remaining ~0.033/round gap is real, not noise — it loses roughly ten times more
per round than correct play does. Blackjack's per-round reward has a standard deviation near 1.15,
which is why these are reported with intervals rather than as bare means; a 2,000-round measurement
of the same agent carries a ±0.05 error bar and can look like anything at all.

That basic strategy measures −0.0034 here is a useful check on the engine: the published house edge
for these rules is around −0.005, and the two agree within the interval.

### Strategy accuracy — and why it reads lower than you might expect

| Metric | Result |
|--------|--------|
| Overall vs basic strategy | 86.2% (448/520 states) |
| — two-card hands | 81.2% (195/240) — doubling lives here |
| — multi-card hands | 90.4% (253/280) — hit/stand only |
| Not scored | 200 states where some legal action was never trained |
| Win / loss / push | 43.0% / 48.2% / 8.8% |
| Training throughput | ~2.9M episodes/sec (5M episodes in ~1.7s) |
| Decision latency | ~3.2 ns (310M decisions/sec) |
| Game simulation | ~7.8M hands/sec |

The audit scores **two policies, not one**. `State::hash()` packs `canDouble` as a bit, so a hand of
11 that can double and a hand of 11 that has already hit occupy different Q-table slots and learn
independently. Scoring only one of them measures half the agent.

States where some legal action never received a single learning update are reported as coverage gaps
rather than scored. An untried action keeps the Q-table's initial 0.0, which beats every genuinely
negative alternative, so its argmax reflects initialisation rather than a decision. The 200 unscored
states are exactly the unreachable ones — soft totals below 13, blackjacks, and pair-only
compositions — which is a useful self-check on the audit.

<p align="center">
  <img src="docs/images/learning_curve.png" alt="Learning curve: outcome rates and strategy accuracy over 5M episodes" width="850">
</p>

Win rate is a poor progress signal in blackjack — it saturates near 43% almost immediately, because
even perfect play loses more hands than it wins.

<p align="center">
  <img src="docs/images/q_heatmap.png" alt="Q-value heatmap over hard totals" width="720">
</p>

<p align="center">
  <em>Best learned Q-value per (player total × dealer upcard) on the opening hand. The agent
  independently rediscovers the shape of basic strategy: hit everything below 12, stand on 17+, and
  treat 12–16 against a strong upcard as the losing territory it is. Rows 4 and 21 are blank because
  neither is reachable as a two-card non-pair hand.</em>
</p>

### Where it still diverges

**The agent has not learned to double.** Basic strategy doubles hard 11 against every dealer upcard;
the agent hits nearly all of them, and its hard 9 and 10 rows are similarly thin. This is the single
largest remaining EV leak and the clearest target for the next round of work.

The cause is visible in the Q-values. DOUBLE pays ±2 where HIT and STAND pay ±1, so it is the
highest-variance action in the space. At α=0.05 with reward only on the terminal step, its estimate
stays noisy long after the others settle: across a single run, doubling 11 scores +1.06 against a 4
and −0.31 against a 10, when it is uniformly strong. Those signs are sampling noise, not policy.

The rest of the divergences cluster on **soft 18** and marginal hands where the top two Q-values sit
within ~0.1 of each other. The convergence report ranks every divergence by that margin, and tags
each with the context it came from, so close calls are distinguishable from real errors.

Run-to-run variance over five runs: overall accuracy 86.0–87.3%, two-card accuracy 81.7–85.0%, EV
−0.033 to −0.043, EV gap recovered 91.5–93.1%. The EV metric is noticeably more stable than
cell-match accuracy, which is a second reason to lead with it — cell-match weights a hard 4 the same
as a hard 16, and one of those comes up constantly.

---

## Features

| Category | What's included |
|----------|----------------|
| **Game engine** | Full casino blackjack — split (one split per round), double down, late surrender, soft aces, dealer hits soft 17, configurable decks |
| **Q-learning agent** | ε-greedy exploration with decay, flat `std::array` Q-table (cache-friendly), bit-packed state hash, per-action visit counts, binary save/load |
| **Training pipeline** | Episode loop, periodic evaluation, early stopping, progress bar, checkpoint saves on SIGINT |
| **Strategy validation** | Exhaustive convergence report vs basic strategy, split by hand context, with coverage reporting |
| **EV baselines** | In-engine basic-strategy and random-play agents; every run reports agent vs book vs random EV with confidence intervals |
| **Color-coded chart** | Terminal strategy grid — green = correct, red = wrong, yellow = uncertain |
| **Interactive play** | `./play --mode human|ai|advisor` — play yourself, watch the AI, or get move-by-move advice |
| **Beginner mode** | Plain-English card explanations, chip balance, AI reasoning in natural language |
| **INI config file** | All hyperparameters in `config/default.cfg` and `config/tuned.cfg`; CLI flags override config |
| **Rule presets** | Vegas Strip, Downtown, Atlantic City, European, Single Deck |
| **Python analysis** | `analysis/plot_training.py` generates 5 plots from training logs |
| **Build hygiene** | `-Werror`, `-march=native` and sanitizers are opt-in CMake options, so a clone builds portably by default |
| **Training report** | Full text report saved to `analysis/training_report.txt` after each run |

---

## Quick Start

All commands below assume you are in `core/`. Executables live in `./build/`.

### 1 — Build

```bash
git clone https://github.com/saintparish4/blackjack-ai.git
cd blackjack-ai/core

# First time or after CMakeLists.txt change
cmake -B build -S .
cmake --build build

# Incremental build (after editing code)
cmake --build build
```

Optional CMake flags, all off by default so a fresh clone builds and runs anywhere:

| Option | Effect |
|--------|--------|
| `-DBLACKJACK_WERROR=ON` | Treat warnings as errors. CI uses this; it is a poor default for consumers on a newer compiler. |
| `-DBLACKJACK_NATIVE_ARCH=ON` | Add `-march=native`. Faster local training; the binary then crashes with SIGILL on an older CPU. |
| `-DBLACKJACK_SANITIZE=ON` | Build with ASan + UBSan. |

`CMAKE_BUILD_TYPE` defaults to `Release` and is respected — a `Debug` build is actually unoptimized.

### 2 — Train

```bash
# 1 million episodes with Vegas Strip rules (default)
./build/train

# Custom episode count
./build/train --episodes 500000

# Load INI config
./build/train --config ../config/default.cfg

# Resume from checkpoint
./build/train --episodes 1000000 --checkpoint ./checkpoints/agent_episode_50000

# Verbose (convergence report, etc.)
./build/train --episodes 10000 --verbose

# Config + episode override
./build/train --config ../config/default.cfg --episodes 5000

# All options
./build/train --help
```

Training artifacts land in the working directory:

| Path | Contents |
|------|----------|
| `models/final_agent` | Saved Q-table + metadata |
| `analysis/q_table.csv` | Q-values **and per-action visit counts** for every visited state, keyed by the full state including `can_split`/`can_double` |
| `analysis/training_report.txt` | Full post-training report |
| `logs/training_YYYYMMDD_HHMMSS.csv` | Per-evaluation metrics (episode, win rate, epsilon, …) |
| `checkpoints/` | Periodic agent snapshots |

### 3 — Play

```bash
# Play yourself (human mode)
./build/play --mode human

# Watch the AI play
./build/play --mode ai --model ./models/final_agent --hands 5

# Get move-by-move AI advice while you play
./build/play --mode advisor --model ./models/final_agent --hands 5

# Beginner mode — plain-English explanations, chip balance
./build/play --mode human --hands 5 --beginner
./build/play --mode advisor --model ./models/final_agent --hands 5 --beginner

# Unlimited hands (prompt "Continue? [Y/n]" after each hand)
./build/play --mode human --hands 0

# Print strategy chart + convergence report for a saved model
./build/play --report --model ./models/final_agent

# All options
./build/play --help
```

Every decision prints the Q-values it was chosen from, so the agent's reasoning is inspectable
rather than a black box:

<p align="center">
  <img src="docs/images/play_ai_mode.png" alt="AI play mode showing per-decision Q-values and a split" width="720">
</p>

<p align="center">
  <em>The agent splits a pair of 3s against a king, then plays each resulting hand to completion
  and settles them independently — one busts, one wins on the dealer bust.</em>
</p>

### 4 — Visualise

Python commands assume you are in the repo root (or the directory containing `analysis/`).

```bash
# Optional: create and activate a virtual environment
python3 -m venv .venv
# Unix:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r analysis/requirements.txt

# Generate all 5 plots from your training logs
python analysis/plot_training.py logs/training_*.csv --output plots/

# Point at a specific Q-table for the heatmap
python analysis/plot_training.py logs/*.csv \
    --qtable analysis/q_table.csv \
    --output plots/
```

Generated plots:

| File | Contents |
|------|----------|
| `learning_curve.png` | Win / loss / push rates and strategy accuracy vs episode, basic-strategy reference line |
| `reward_curve.png` | Average reward per evaluation checkpoint |
| `epsilon_decay.png` | Exploration rate decay over the training run |
| `state_coverage.png` | Number of Q-table states visited over time |
| `q_heatmap.png` | Best Q-value per (player total × dealer up-card) for hard totals |

### 5 — Test

From `core/`:

```bash
./build/run_tests

# Or with ctest
cd build && ctest --output-on-failure

# Run a single test suite directly (faster)
./build/run_tests --gtest_filter="QLearningTest.*"
./build/run_tests --gtest_filter="TrainerTest.*"
```

114 tests across 6 suites, run on every push against both GCC and Clang (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)). CI builds with `-Wall -Wextra -Werror
-pedantic`, runs a separate ASan + UBSan job, and smoke-tests the full train → evaluate → report →
save pipeline, which the unit tests only cover in pieces.

Tests that sample random deals are sized so a spurious failure is vanishingly unlikely, and they
account for the cases that make blackjack assertions subtle — a round can settle on the deal before
the agent ever acts, and simultaneous blackjacks push rather than pay.

---

## Configuration

Two configs ship with the project: `config/default.cfg` (fast, general-purpose) and
`config/tuned.cfg` (the settings behind the numbers above). Both document every knob inline.

```ini
# Training
episodes             = 1000000
eval_frequency       = 10000
eval_games           = 1000
checkpoint_frequency = 50000

# Q-Learning
learning_rate   = 0.1
discount_factor = 0.95
epsilon         = 1.0
epsilon_decay   = 0.99995
epsilon_min     = 0.01

# Game rules (preset or per-field overrides)
rules_preset         = vegas-strip
# num_decks          = 6
# dealer_hits_soft_17 = false
# surrender          = false
```

CLI flags override config values; config values override built-in defaults. A flag only overrides
when it is actually passed — a flag's own default never silently outranks the config file.

### Rule Presets

| Preset | Decks | Dealer S17 | Surrender |
|--------|-------|------------|-----------|
| `vegas-strip` | 6 | Stands | No |
| `downtown` | 2 | Hits | Yes |
| `atlantic-city` | 8 | Stands | Yes |
| `european` | 6 | Stands | No |
| `single-deck` | 1 | Hits | No |

---

## Architecture

```
blackjack-ai/
├── .github/workflows/     # CI: GCC + Clang build, tests, pipeline smoke test
├── core/
│   ├── include/
│   │   ├── game/          # Card, Deck, Hand, BlackjackGame, GameRules
│   │   ├── ai/            # QLearningAgent, State, PolicyTable, GameStateConverter
│   │   ├── training/      # Trainer, Evaluator, Logger, ConvergenceReport, StrategyChart
│   │   └── util/          # ArgParser, ConfigParser, ProgressBar
│   ├── scripts/           # train.cpp, play.cpp, benchmark.cpp
│   └── tests/             # 114 unit tests (Google Test)
├── analysis/
│   ├── plot_training.py   # Python visualisation script
│   └── requirements.txt   # matplotlib, pandas
├── config/
│   ├── default.cfg        # INI config template
│   └── tuned.cfg          # Settings behind the published results
└── docs/
    ├── images/            # Plots and terminal captures used in this README
    └── training_report.txt # Full report from the published run
```

Three static libraries are linked together: `blackjack_game` → `blackjack_ai` → `blackjack_training`.

### Layer 1 — Game Engine

- **`Card`, `Deck`, `Hand`** — primitive types. `Hand::getValue()` returns `{total, isSoft}` with cached result (invalidated on mutation). `Deck` accepts an optional seed for deterministic tests.
- **`BlackjackGame`** — single-player vs dealer. Supports split (one split per round, sequential hands), double down, late surrender, and immediate-blackjack detection. `getOutcomes()` / `getWasDoubledByHand()` return one entry per hand.
- **`GameRules`** — house rules struct with static preset factories.

### Layer 2 — AI

- **`State`** — discrete RL state: `{playerTotal, dealerUpCard, hasUsableAce, canSplit, canDouble}`. Bit-packed via `hash()` for O(1) Q-table lookup. Note that `canSplit`/`canDouble` are part of the key: the same total occupies a different slot depending on what is legal there, so the opening-hand and already-hit policies are learned independently.
- **`Action`** — `HIT`, `STAND`, `DOUBLE`, `SPLIT`, `SURRENDER`.
- **`QLearningAgent`** — Q(s,a) ← Q + α[R + γ max Q(s′,a′) − Q]; ε-greedy with decay. Flat `std::array<QValues, 4096>` Q-table; binary save/load; CSV export.
- **`PolicyTable`** — the Q-table itself, plus a per-action visit count. The count is what separates "the agent evaluated this action and disliked it" from "this action was never legal here, so its Q is still 0.0"; both leave a Q-value, only the first is a decision, and treating them alike lets an untrained action win a greedy argmax.
- **`GameStateConverter`** — converts game state → AI state, enumerates valid actions, executes chosen action.

### Layer 3 — Training

- **`Trainer`** — episode loop, periodic evaluation, progress bar, early stopping, checkpoint saves. Runs the convergence report and saves `analysis/training_report.txt` at the end of every `train()` call.
- **`Evaluator`** — exploitation-mode evaluation, reward standard error, and the shared `legalActionsFor` / `buildAuditStates` definitions used by every audit and chart.
- **`BasicStrategyAgent` / `RandomAgent`** — reference policies implementing `ai::Agent`, so they run through the identical evaluator, engine and rules as a learned agent. A baseline computed any other way is not comparable.
- **`ConvergenceReport`** — exhaustive policy audit over both hand contexts: all `(playerTotal 4–21) × (dealerCard 1–10) × (soft/hard)` states, scored separately for two-card and multi-card hands, divergences sorted by Q-value margin, coverage gaps reported rather than scored.
- **`StrategyChart`** — colour-coded terminal grid (green / red / yellow per cell).
- **`Logger`** — writes CSV training logs to disk.

### Reward Design

| Outcome | Base reward | With double-down |
|---------|-------------|-----------------|
| Win / dealer bust | +1.0 | +2.0 |
| Blackjack | +1.5 | — |
| Push | 0.0 | 0.0 |
| Loss / bust | −1.0 | −2.0 |
| Surrender | −0.5 | — |

Only the terminal experience in each episode carries a non-zero reward. All intermediate experiences get `0.0`.

---

## Reproducing the numbers

```bash
cd core
./build/train --config ../config/tuned.cfg     # ~4s, writes models/ logs/ analysis/

# Regenerate the plots in docs/images/
cd .. && pip install -r analysis/requirements.txt
python analysis/plot_training.py core/logs/training_*.csv \
    --qtable core/analysis/q_table.csv --output docs/images/
```

`config/default.cfg` converges faster but plateaus lower: early stopping keys on win rate, which
flattens long before the policy settles on marginal hands. `config/tuned.cfg` lowers the learning
rate, holds a floor under epsilon so rare soft totals keep accumulating visits, and raises the
early-stopping patience accordingly. Both files document every knob inline.

The agent is seeded non-deterministically, so numbers move between runs. Across five runs: overall
accuracy 86.0–87.3%, two-card accuracy 81.7–85.0%, EV −0.033 to −0.043, EV gap recovered
91.5–93.1%.

---

## Benchmark

From `core/`:

```bash
./build/benchmark
./build/benchmark --games 50000 --decisions 500000
./build/benchmark --help
```

Measures game simulation throughput and per-decision Q-lookup latency independently.

---

## Future Work

Ordered by what the measurements above actually justify.

- **Close the double-down gap** — the agent's largest EV leak. DOUBLE pays ±2 against ±1 for
  hit/stand, so its estimate stays noise-dominated at a uniform learning rate. Candidates: a
  visit-count-scaled α (the counts are already tracked), reward normalisation, or optimistic
  initialisation restricted to legal actions. Success criterion is EV, not cell-match: close the
  0.033/round gap to basic strategy.
- **Expected SARSA** — an on-policy comparison run through the same evaluator and baselines. Cheap
  now that the harness exists, and the expectation over next actions should cut exactly the variance
  that is hurting DOUBLE.
- **Hi-Lo card counting** — extend the state space with a running count, then measure the EV gain
  against the flat-bet baseline. Counting is the one blackjack strategy that beats basic strategy,
  so this is the natural ceiling-raiser.
- **Deep Q-Network (DQN)** — neural function approximation, once the tabular case is actually solved.
- **WebAssembly browser demo** — compile the engine plus a trained model to WASM and play in-browser.
- **Multi-player tables** — extend `BlackjackGame` to track multiple player seats.
- **SIMD-optimized parallel evaluation** — vectorised hand simulation for faster batch training.

---

## License

MIT

---

Built by Sharif Parish · C++ / Reinforcement Learning / Open to collaborations
