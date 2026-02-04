# Blackjack AI

> **Experimental** — An intelligent Blackjack platform combining C++ performance, reinforcement learning, real-time multiplayer, and future AR capabilities.

## What Makes This Different

| Feature | Description |
|---------|-------------|
| **RL-Powered AI** | Reinforcement learning agent trained for optimal play, compiled to WebAssembly for browser deployment |
| **Real-Time Multiplayer** | Synchronized gameplay with AI dealer managing multiple concurrent players |
| **AR Card Detection** | Smart glasses integration for real-world card recognition *(future planning)* |

## Architecture

```
blackjack-ai/
├── core/                    # C++ game engine
│   ├── include/game/        # Core game logic
│   │   ├── Card.hpp/cpp     # Card representation
│   │   ├── Deck.hpp/cpp     # Deck management & shuffling
│   │   ├── Hand.hpp/cpp     # Hand evaluation & scoring
│   │   ├── BlackjackGame.*  # Game state machine
│   │   └── GameRules.hpp    # Configurable rule variants
│   └── tests/               # Unit tests
├── ai/                      # Reinforcement learning (planned)
│   ├── agents/              # RL agent implementations
│   ├── training/            # Training infrastructure
│   └── models/              # Trained model artifacts
├── server/                  # Multiplayer backend (planned)
└── web/                     # WebAssembly frontend (planned)
```

## Tech Stack

- **Core Engine**: C++17 with CMake
- **AI**: Reinforcement Learning (Q-Learning / Deep RL)
- **Deployment**: WebAssembly via Emscripten
- **Multiplayer**: WebSocket-based real-time sync
- **AR**: OpenCV + custom detection pipeline *(future)*

## Building

### Prerequisites

- CMake 3.16+
- C++17 compatible compiler
- Google Test (fetched automatically)

### Build Commands

```bash
cd core
mkdir build && cd build
cmake ..
cmake --build .
```

### Run Tests

```bash
cd core/build
ctest --output-on-failure
```

## Roadmap

- [x] Core game engine (Card, Deck, Hand, Game logic)
- [ ] Complete game rules & variant support
- [ ] RL agent training pipeline
- [ ] WebAssembly compilation
- [ ] Multiplayer server infrastructure
- [ ] Web client interface
- [ ] AR card detection module

## Project Status

**Stage**: Early Development / Experimental

This is an experimental project exploring the intersection of classical game AI, modern reinforcement learning, and emerging AR technologies. Expect breaking changes and rapid iteration.

## License

MIT
