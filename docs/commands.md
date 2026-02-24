
---

# [   BUILD & C++   ]

Two workflows: run everything from `core/`, or run everything from `core/build/`. Pick one and use its section.

---

## Workflow 1 — Start from `cd core/`

Stay in `core/` for build and run. Executables live in `./build/`.

### Build (first time or after CMakeLists.txt change)
- cd core
- cmake -B build -S .
- cmake --build build

### Incremental build (after editing code)
- cmake --build build

### Clean build (optional)
- rm -rf build

### Tests
- ./build/run_tests
- cd build && ctest --output-on-failure

### Train
- ./build/train --help     [ prints usage with all flags ]
- ./build/train
- ./build/train --episodes 500000
- ./build/train --episodes 1000000 --checkpoint ./checkpoints/agent_episode_50000
- ./build/train --episodes 10000 --verbose
- ./build/train --config ../config/default.cfg
- ./build/train --config ../config/default.cfg --episodes 5000

### Benchmark
- ./build/benchmark --help
- ./build/benchmark
- ./build/benchmark --games 50000 --decisions 500000

### Play
- ./build/play --help
- ./build/play --report --model ./models/final_agent
- ./build/play --mode human --hands 5
- ./build/play --mode human --hands 5 --beginner
- ./build/play --mode human --hands 0
- ./build/play --mode ai --model ./models/final_agent --hands 5
- ./build/play --mode ai --model ./models/final_agent --hands 10 --beginner
- ./build/play --mode advisor --model ./models/final_agent --hands 5
- ./build/play --mode advisor --model ./models/final_agent --hands 5 --beginner

### After adding/changing a source in CMakeLists.txt
- cmake -B build -S .
- cmake --build build

---

#### Note 
- `--hands 0` = unlimited hands with "Continue? [Y/n]" after each hand.

---

# Commands reference

Python commands assume you are in `analysis/` (or the directory containing `requirements.txt`).


# [   PYTHON   ]

## Create virtual environment
- python3 -m venv .venv

### Activate (Unix)
- source .venv/bin/activate

### Activate (Windows PowerShell)
- .venv\Scripts\Activate.ps1

## Install dependencies
- pip install -r requirements.txt
