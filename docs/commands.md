
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
- ./build/train --config ../config/default.cfg --seed 42     [ reproducible run ]
- ./build/train --config ../config/tuned.cfg                [ the published run, seed 1 ]

### Experiment (one config over N seeds)
- ./build/experiment --help
- ./build/experiment --config ../config/tuned.cfg --seeds 5
- ./build/experiment --config ../config/default.cfg --seeds 3 --episodes 200000 --out ../experiments --name my-sweep
- ./build/experiment --config ../config/tuned.cfg --seeds 10 --seed-base 100 --threads 4

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
- ./build/play --mode ai --model ./models/final_agent --hands 2 --seed 1     [ replay a shoe ]
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

## Publish results (from the repo root)
- analysis/publish_results.sh     [ retrains tuned.cfg (seed 1), rewrites docs/training_report.txt and docs/images/ ]
