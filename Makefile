# Convenience front end for the CMake build, the tests, the CLI tools, the
# browser demo and the experiment scripts. CMake stays the real build system;
# every target here is a thin wrapper you could type by hand.
#
#   make            show this help
#   make build      configure (once) and build everything in core/build
#   make test       build, then run the unit tests
#
# Override variables on the command line, e.g.
#   make build BUILD_TYPE=Debug CXX=clang++
#   make train ARGS="--config ../config/tuned.cfg"
#   make test FILTER='Hand*'

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CORE       := $(ROOT)/core
BUILD      ?= $(CORE)/build
ASAN_BUILD ?= $(CORE)/build-asan
BUILD_TYPE ?= Release
JOBS       ?= $(shell nproc 2>/dev/null || echo 4)
WERROR     ?= OFF
NATIVE     ?= OFF
ARGS       ?=
CMAKE_EXTRA ?=
FILTER     ?= *
CXX        ?= g++
CTEST_JOBS ?= $(JOBS)
# Rule sets for compare-solver, comma-separated, each "DECKS H17 DAS SURRENDER".
# The default is the five CI checks.
RULES      ?= 6 n n n,6 y y n,1 y n n,2 y n y,8 n n y

# Ninja when it is installed, CMake's default otherwise. An existing build
# directory keeps whatever generator it was created with.
GENERATOR  ?= $(if $(shell command -v ninja 2>/dev/null),-G Ninja,)

PYTHON     ?= $(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3)
PORT       ?= 8000

CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
              -DCMAKE_CXX_COMPILER=$(CXX) \
              -DBLACKJACK_WERROR=$(WERROR) \
              -DBLACKJACK_NATIVE_ARCH=$(NATIVE) \
              $(CMAKE_EXTRA)

SAN_ENV = ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
          UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

.PHONY: help configure reconfigure build debug native rebuild \
        test test-verbose ctest asan test-asan smoke ci \
        train play report benchmark experiment syntax \
        solve farmer compare-solver \
        wasm serve web-test venv publish exp001 exp002 \
        clean clean-artifacts distclean

## ---- Build ---------------------------------------------------------------

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} \
	  /^## ----/ {sub(/^## -+ */, ""); sub(/ -+$$/, ""); printf "\n\033[1m%s\033[0m\n", $$0} \
	  /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Variables: BUILD_TYPE=$(BUILD_TYPE) CXX=$(CXX) WERROR=$(WERROR) NATIVE=$(NATIVE) JOBS=$(JOBS)"
	@echo "           ARGS=\"...\" passes flags to train/play/benchmark/experiment"

$(BUILD)/CMakeCache.txt:
	cmake -S $(CORE) -B $(BUILD) $(GENERATOR) $(CMAKE_FLAGS)

configure: $(BUILD)/CMakeCache.txt ## Configure core/build (only if not yet configured)

reconfigure: ## Re-run CMake with the current variables (e.g. after changing BUILD_TYPE)
	cmake -S $(CORE) -B $(BUILD) $(CMAKE_FLAGS)

build: configure ## Build all targets (libs, tests, train, play, benchmark, experiment)
	cmake --build $(BUILD) --parallel $(JOBS)

debug: ## Debug build in core/build-debug
	$(MAKE) build BUILD=$(CORE)/build-debug BUILD_TYPE=Debug

native: ## Release build with -march=native (fast, not portable) in core/build-native
	$(MAKE) build BUILD=$(CORE)/build-native NATIVE=ON

rebuild: ## Wipe core/build and build from scratch
	rm -rf $(BUILD)
	$(MAKE) build

## ---- Test ----------------------------------------------------------------

test: build ## Run unit tests (FILTER='Suite.*' to narrow)
	$(BUILD)/run_tests --gtest_filter='$(FILTER)' --gtest_brief=1

test-verbose: build ## Run unit tests with full gtest output
	$(BUILD)/run_tests --gtest_filter='$(FILTER)'

ctest: build ## Run the tests through ctest in parallel, as CI does (CTEST_JOBS=1 for serial)
	ctest --test-dir $(BUILD) --output-on-failure --parallel $(CTEST_JOBS)

asan: ## Build with address + UB sanitizers in core/build-asan
	$(MAKE) build BUILD=$(ASAN_BUILD) BUILD_TYPE=Debug WERROR=ON \
	  CMAKE_EXTRA=-DBLACKJACK_SANITIZE=ON

test-asan: asan ## Unit tests and a short training run under sanitizers
	$(SAN_ENV) ctest --test-dir $(ASAN_BUILD) --output-on-failure --parallel $(CTEST_JOBS)
	cd $(CORE) && $(SAN_ENV) $(ASAN_BUILD)/train --episodes 5000 >/dev/null
	@echo "sanitizer run clean"

smoke: build ## CI's end-to-end smoke tests (train, report, deterministic experiments)
	@tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; cd $(CORE); \
	echo "== train pipeline"; \
	$(BUILD)/train --episodes 20000 >/dev/null; \
	test -f models/final_agent.qtable; test -f analysis/q_table.csv; \
	echo "== strategy report"; \
	$(BUILD)/play --report --model ./models/final_agent >/dev/null; \
	echo "== experiment determinism"; \
	for run in a b; do \
	  $(BUILD)/experiment --config ../config/default.cfg --seeds 2 \
	    --episodes 20000 --eval-rounds 20000 --out $$tmp/exp-$$run --name smoke \
	    | grep -v '^Output\|^Wrote' > $$tmp/exp-$$run.txt; \
	done; \
	diff $$tmp/exp-a/smoke/results.csv $$tmp/exp-b/smoke/results.csv; \
	diff $$tmp/exp-a.txt $$tmp/exp-b.txt; \
	echo "== card counting determinism"; \
	printf 'extends = %s/../config/default.cfg\nmax_bet = 8\nuse_count = true\nlearn_bets = true\nbet_calibration_rounds = 20000\n' "$$PWD" > $$tmp/count.cfg; \
	for run in a b; do \
	  $(BUILD)/experiment --config $$tmp/count.cfg --seeds 2 --episodes 20000 \
	    --eval-rounds 20000 --out $$tmp/count-$$run --name smoke >/dev/null; \
	done; \
	diff $$tmp/count-a/smoke/results.csv $$tmp/count-b/smoke/results.csv; \
	diff $$tmp/count-a/smoke/count_profile.csv $$tmp/count-b/smoke/count_profile.csv; \
	echo "smoke tests passed"

ci: ## Everything CI runs: -Werror build, ctest, smoke tests, sanitizers
	$(MAKE) reconfigure build ctest smoke WERROR=ON BUILD_TYPE=Release
	$(MAKE) test-asan

syntax: ## Syntax-check every header and source under core/include (DIR=solver or DIR=solver/Shoe.hpp)
	@cd $(CORE); fail=0; \
	for f in $$(find include/$(DIR) -name '*.hpp' -o -name '*.cpp' | sort); do \
	  case $$f in \
	    *.hpp) echo "#include \"$${f#include/}\"" | $(CXX) -std=c++17 -fsyntax-only -Wall -Wextra -pedantic -Iinclude -x c++ - || { echo "FAIL $$f"; fail=1; } ;; \
	    *.cpp) $(CXX) -std=c++17 -fsyntax-only -Wall -Wextra -pedantic -Iinclude $$f || { echo "FAIL $$f"; fail=1; } ;; \
	  esac; \
	done; \
	[ $$fail -eq 0 ] && echo "syntax OK: include/$(DIR)"

## ---- Run -----------------------------------------------------------------
# The binaries write models/, logs/ and analysis/ into the working directory,
# so they run from core/ (the layout docs/commands.md uses).

train: build ## Train an agent (ARGS="--config ../config/tuned.cfg")
	cd $(CORE) && $(BUILD)/train $(ARGS)

play: build ## Play interactively (ARGS="--mode ai --model ./models/final_agent")
	cd $(CORE) && $(BUILD)/play $(if $(ARGS),$(ARGS),--mode human --hands 5)

report: build ## Print the strategy report for core/models/final_agent
	cd $(CORE) && $(BUILD)/play --report --model ./models/final_agent $(ARGS)

benchmark: build ## Run the benchmark (ARGS="--games 50000")
	cd $(CORE) && $(BUILD)/benchmark $(ARGS)

experiment: build ## Run one config over N seeds (ARGS="--config ../config/tuned.cfg --seeds 5")
	cd $(CORE) && $(BUILD)/experiment $(ARGS)

## ---- Solver & external reference ----------------------------------------
# Farmer's CA is GPL-3.0: run_farmer.sh clones it at a pinned commit into
# tools/farmer/src (gitignored) and builds it into tools/farmer/bin. Scripts
# run through bash, so they work whether or not git kept their exec bit.

solve: build ## Exact solver: strategy, EV by upcard, dealer table (ARGS="--decks 1 --h17")
	cd $(CORE) && $(BUILD)/solve $(ARGS)

farmer: ## Fetch and build Farmer's CA into tools/farmer/bin (once)
	bash $(ROOT)/tools/farmer/run_farmer.sh --build-only

compare-solver: build farmer ## Diff solve against Farmer's CA (RULES="6 y y n"; default: all five)
	@cd $(CORE); status=0; sets='$(RULES)'; IFS=,; \
	for rules in $$sets; do \
	  IFS=' '; echo "== $$rules"; \
	  bash $(ROOT)/tools/farmer/compare_solver.sh $$rules || status=1; \
	done; exit $$status

## ---- Web demo ------------------------------------------------------------

wasm: ## Build the WebAssembly module into web/wasm/ (needs Emscripten)
	bash $(ROOT)/web/build.sh

serve: ## Serve the browser demo at http://localhost:8000 (PORT=...)
	python3 -m http.server -d $(ROOT)/web $(PORT)

web-test: ## Node smoke test of the browser demo (run `make wasm` first)
	node $(ROOT)/web/test/smoke.mjs

## ---- Analysis & experiments ----------------------------------------------

venv: $(ROOT)/.venv/bin/python ## Create .venv with analysis/requirements.txt

$(ROOT)/.venv/bin/python: $(ROOT)/analysis/requirements.txt
	python3 -m venv $(ROOT)/.venv
	$(ROOT)/.venv/bin/pip install -r $(ROOT)/analysis/requirements.txt
	@touch $@

publish: build ## Retrain tuned.cfg and regenerate docs/ images, report and web model
	PYTHON=$(PYTHON) bash $(ROOT)/analysis/publish_results.sh $(ARGS)

exp001: build ## Rerun experiment 001 (double down): explore + confirm
	bash $(ROOT)/experiments/001-double-down/run.sh $(if $(ARGS),$(ARGS),all)

exp002: build ## Rerun experiment 002 (card counting): ARGS=explore|confirm|all
	bash $(ROOT)/experiments/002-card-counting/run.sh $(if $(ARGS),$(ARGS),all)

## ---- Clean ---------------------------------------------------------------

clean: ## Remove compiled objects but keep the CMake configuration and googletest
	-cmake --build $(BUILD) --target clean

clean-artifacts: ## Remove training output in core/ (models, logs, checkpoints, analysis)
	rm -rf $(CORE)/models $(CORE)/logs $(CORE)/checkpoints $(CORE)/analysis

distclean: clean-artifacts ## Remove every build directory and training output
	rm -rf $(CORE)/build $(CORE)/build-debug $(CORE)/build-native $(ASAN_BUILD) \
	       $(ROOT)/build-wasm $(ROOT)/web/wasm
