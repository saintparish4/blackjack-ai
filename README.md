# Blackjack AI

High-performance C++ Blackjack AI using Q-Learning reinforcement learning. Currently training to beat basic strategy via self-play—processes millions of hands/second. Early results: 40.0% win rate after 170k episodes (vs basic strategy's ~43%).

## Features

- **Blazing-Fast C++**: ~4.3M games/sec, ~28.7M Q-Learning decisions/sec via cache-friendly data structures
- **Q-Learning Agent**: Self-play training on discretized states (player sum, dealer card, usable ace)
- **Modular Architecture**: Easy to extend with SARSA, DQN, or Monte Carlo Tree Search
- **Benchmark Mode**: Compare against random play and basic strategy
- **Analysis Ready**: Exports Q-table CSV for Python/Matplotlib analysis

## Tech Stack

- C++17 (performance-critical core)
- CMake (cross-platform builds)
- Python + Matplotlib (optional visualization of exported data)
- STL (vectors, algorithms, memory optimization)

## Quick Start

### Clone & Build

```bash
git clone https://github.com/saintparish4/blackjack-ai.git
cd blackjack-ai/core
mkdir build && cd build
cmake ..
cmake --build .
```

### Train

```bash
# 100k episodes (default: 1M)
./train 100000

# Resume from checkpoint
./train 500000 ./checkpoints/checkpoint_50000
```

### Benchmark

```bash
./benchmark
```

### Run Tests

```bash
ctest --output-on-failure
```

## Outputs

| Output | Location |
|--------|----------|
| Trained model | `models/final_agent` |
| Q-table (CSV) | `analysis/q_table.csv` |
| Training logs | `logs/` |
| Checkpoints | `checkpoints/` |

## Current Progress

| Strategy | Win % (1k eval games) | Notes |
|----------|------------------------|------|
| Random play | ~33% | Baseline |
| Basic strategy | ~43% | Target reference |
| Q-Learning | 40.0% | 170k episodes, early stopping |

Rules: 6 decks, dealer hits soft 17, blackjack 3:2.

| Metric | Result |
|--------|--------|
| Strategy accuracy vs basic | 83.2% |
| States learned | 589 |
| Game simulation | ~4.3M games/sec |
| Q-Learning decisions | ~28.7M decisions/sec |
| Decision latency | ~35 ns/decision |

Full convergence expected at 1M+ episodes.

## Architecture

```
States: Discrete (player sum 4–21 × dealer card 2–A × usable ace Y/N)
Actions: Hit/Stand (expandable to Double/Split)
Q-Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
Exploration: ε-greedy (1.0 → 0.01 decay)
```

**C++ performance notes:**

- Contiguous Q-table arrays (no allocations mid-training)
- 50x faster than equivalent Python

## Project Structure

```
blackjack-ai/
├── core/
│   ├── include/
│   │   ├── game/          # Card, Deck, Hand, BlackjackGame
│   │   ├── ai/            # QLearningAgent, State, PolicyTable
│   │   └── training/      # Trainer, Evaluator, Logger
│   ├── scripts/           # train.cpp, benchmark.cpp
│   └── tests/             # Unit tests
```

## Next Milestones

- [ ] Full convergence testing (1M+ episodes)
- [ ] Multi-deck shoe + Hi-Lo card counting
- [ ] Deep Q-Network with CUDA
- [ ] Web demo via WebAssembly
- [ ] Unit tests + continuous integration
- [ ] SIMD-optimized parallel hand evaluation
- [ ] Python/Matplotlib visualization script (plot_results.py)

## Why This Matters

Demonstrates C++ systems programming + reinforcement learning—a useful combo for game dev, trading algos, or real-time AI. Scales from toy problem to production-grade decision engine.

## License

MIT 

---

Built by Sharif Parish · Open to C++/AI/ML collaborations · WIP—metrics updating weekly.
