
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

### Exact solver and grading
- ./build/solve --help
- ./build/solve     [ optimal strategy, EV by upcard and dealer table: 6D S17, the engine's rules ]
- ./build/solve --decks 1 --h17
- ./build/solve --policy book     [ the repo's basic strategy chart, exactly ]
- ./build/audit --help
- ./build/audit --preset downtown --rounds 1000000
- ./build/audit --rounds 20000000 --cut-card-rounds 20000000     [ experiment 003's full run ]
- ./build/regret --models ../web/model/agent --top 10     [ exact EV lost, itemised by state ]
- ./build/regret --models ../web/model/agent --export-farmer /tmp/published
- ../tools/farmer/bin/policy_ev 6 n n n /tmp/published-0.policy     [ Farmer's code, same EV ]
- ./build/indices --config ../experiments/005-index-plays/arms/linear.cfg --theory
- ./build/indices --config ../experiments/005-index-plays/arms/linear.cfg --seeds 2 --checkpoints 1e6,1e7

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
- ./build/play --mode ai --model ./models/final_agent --hands 3 --seed 59    [ replay a shoe ]
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

## Card counting (from the repo root)
- experiments/002-card-counting/run.sh explore     [ every arm on seeds 1-5 ]
- experiments/002-card-counting/run.sh confirm     [ pre-registered arms on fresh seeds 101-110 ]
- python analysis/leaderboard.py experiments/002-card-counting/confirm
- python analysis/plot_counting.py experiments/002-card-counting/confirm/<arm> -o docs/images/counting_edge.png

## Exact results (from the repo root)
- make compare-solver     [ the solver against Farmer's CA, all five CI rule sets; fetches it once ]
- make compare-solver RULES="6 y y n"
- make audit ARGS="--rounds 20000000 --cut-card-rounds 20000000"
- make exp005 ARGS=theory     [ what the effects of removal predict, ~25 s ]
- make exp005 ARGS=explore    [ every arm, seeds 1-5; ~30 min on 8 cores ]
- make exp005 ARGS=confirm    [ candidates.txt, seeds 101-110; ~26 min on 8 cores ]

## Browser demo (from the repo root)
- web/build.sh                          [ needs Emscripten; writes web/wasm/ ]
- python3 -m http.server -d web 8000    [ open http://localhost:8000, add ?seed=N to replay a shoe ]
- node web/test/smoke.mjs               [ 20,000 rounds through the page's API ]

## Rank experiment arms (from the repo root)
- python analysis/leaderboard.py experiments/001-double-down/confirm
- python analysis/leaderboard.py experiments/001-double-down/explore --markdown
- experiments/001-double-down/run.sh all     [ reruns experiment 001: explore + confirm ]

## Publish results (from the repo root)
- analysis/publish_results.sh     [ retrains tuned.cfg (seed 1), rewrites docs/training_report.txt and docs/images/ ]
