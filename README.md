# Blackjack AI

A  C++ Q-learning agent that learns optimal blackjack strategy through self-play.
Trains at millions of episodes per second, converges to basic strategy accuracy above 90%, and ships
with an interactive play mode, a color-coded strategy chart, Python visualisation scripts, and a
full post-training report.

---

## Features

| Category | What's included |
|----------|----------------|
| **Game engine** | Full casino blackjack — split (one split per round), double down, late surrender, soft aces, dealer hits soft 17, configurable decks |
| **Q-learning agent** | ε-greedy exploration with decay, flat `std::array` Q-table (cache-friendly), bit-packed state hash, binary save/load |
| **Training pipeline** | Episode loop, periodic evaluation, early stopping, progress bar, checkpoint saves on SIGINT |
| **Strategy validation** | Exhaustive convergence report vs basic strategy after every training run |
| **Color-coded chart** | Terminal strategy grid — green = correct, red = wrong, yellow = uncertain |
| **Interactive play** | `./play --mode human|ai|advisor` — play yourself, watch the AI, or get move-by-move advice |
| **Beginner mode** | Plain-English card explanations, chip balance, AI reasoning in natural language |
| **INI config file** | All hyperparameters in `config/default.cfg`; CLI flags override config |
| **Rule presets** | Vegas Strip, Downtown, Atlantic City, European, Single Deck |
| **Python analysis** | `analysis/plot_training.py` generates 5 plots from training logs |
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
| `analysis/q_table.csv` | Q-values for every visited state (for Python heatmap) |
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
| `learning_curve.png` | Win / loss / push rates vs episode, basic-strategy reference line |
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

---

## Configuration

All knobs live in `config/default.cfg`:

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

CLI flags override config values; config values override built-in defaults.

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
├── core/
│   ├── include/
│   │   ├── game/          # Card, Deck, Hand, BlackjackGame, GameRules
│   │   ├── ai/            # QLearningAgent, State, PolicyTable, GameStateConverter
│   │   ├── training/      # Trainer, Evaluator, Logger, ConvergenceReport, StrategyChart
│   │   └── util/          # ArgParser, ConfigParser, ProgressBar
│   ├── scripts/           # train.cpp, play.cpp, benchmark.cpp
│   └── tests/             # Unit tests (Google Test)
├── analysis/
│   ├── plot_training.py   # Python visualisation script
│   └── requirements.txt   # matplotlib, pandas
└── config/
    └── default.cfg        # INI config template
```

Three static libraries are linked together: `blackjack_game` → `blackjack_ai` → `blackjack_training`.

### Layer 1 — Game Engine

- **`Card`, `Deck`, `Hand`** — primitive types. `Hand::getValue()` returns `{total, isSoft}` with cached result (invalidated on mutation). `Deck` accepts an optional seed for deterministic tests.
- **`BlackjackGame`** — single-player vs dealer. Supports split (one split per round, sequential hands), double down, late surrender, and immediate-blackjack detection. `getOutcomes()` / `getWasDoubledByHand()` return one entry per hand.
- **`GameRules`** — house rules struct with static preset factories.

### Layer 2 — AI

- **`State`** — discrete RL state: `{playerTotal, dealerUpCard, hasUsableAce, canSplit, canDouble}`. Bit-packed via `hash()` for O(1) Q-table lookup.
- **`Action`** — `HIT`, `STAND`, `DOUBLE`, `SPLIT`, `SURRENDER`.
- **`QLearningAgent`** — Q(s,a) ← Q + α[R + γ max Q(s′,a′) − Q]; ε-greedy with decay. Flat `std::array<QValues, 4096>` Q-table; binary save/load; CSV export.
- **`GameStateConverter`** — converts game state → AI state, enumerates valid actions, executes chosen action.

### Layer 3 — Training

- **`Trainer`** — episode loop, periodic evaluation, progress bar, early stopping, checkpoint saves. Runs the convergence report and saves `analysis/training_report.txt` at the end of every `train()` call.
- **`Evaluator`** — exploitation-mode evaluation. `BasicStrategy` reference for accuracy comparison.
- **`ConvergenceReport`** — exhaustive policy audit: all `(playerTotal 4–21) × (dealerCard 1–10) × (soft/hard)` states, divergences sorted by Q-value margin, critical-state flags.
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

## Sample Results

After 1 million episodes (Vegas Strip rules, default hyperparameters):

| Metric | Result |
|--------|--------|
| Win rate | ~42–43% |
| Strategy accuracy vs basic | ≥ 90% |
| States explored | ~300–400 |
| Training speed | ~4–5 M episodes / sec |
| Decision latency | ~35 ns / decision |

The convergence report printed at the end of training shows which states still diverge from basic strategy and ranks them by Q-value confidence.

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

- **Hi-Lo card counting** — extend the state space with a running count; train a counting-aware agent
- **SARSA / Expected SARSA** — add alternative on-policy agents for empirical comparison
- **Deep Q-Network (DQN)** — neural function approximation for continuous or higher-dimensional state spaces
- **WebAssembly browser demo** — compile the game engine + a pre-trained model to WASM; run in-browser
- **Multi-player tables** — extend `BlackjackGame` to track multiple player seats
- **SIMD-optimized parallel evaluation** — vectorised hand simulation for faster batch training
- **Card counting integration test** — measure actual EV gain from counting vs flat-bet baseline

---

## License

MIT

---

Built by Sharif Parish · C++ / Reinforcement Learning / Open to collaborations
