cmake_minimum_required(VERSION 3.15)
project(BlackjackAI VERSION 1.0 LANGUAGES CXX)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Skip timestamp checks to avoid clock skew warnings (useful for WSL)
set(CMAKE_SKIP_TIMESTAMP_CHECK ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Threading (used by training library). The browser build has no threads.
if(NOT EMSCRIPTEN)
    find_package(Threads REQUIRED)
endif()

# A plain `cmake -B build` should produce an optimized build rather than the
# unoptimized default, but without hard-coding flags that override the build
# type the user actually asked for.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# Warnings-as-errors is right for CI and hostile as a default: a consumer on a
# newer compiler gets a build failure from a warning that has nothing to do
# with them. CI turns it on explicitly.
option(BLACKJACK_WERROR "Treat compiler warnings as errors" OFF)

# -march=native bakes in the build machine's instruction set. It is a real
# speedup for local training runs and makes the binary crash with SIGILL on any
# older CPU, so it cannot be on by default in a repo people clone.
option(BLACKJACK_NATIVE_ARCH "Optimize for the building machine's CPU" OFF)

# Sanitizers, for the CI job that catches what -Wall cannot.
option(BLACKJACK_SANITIZE "Build with address and UB sanitizers" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic)

    if(BLACKJACK_WERROR)
        add_compile_options(-Werror)
    endif()

    # CMAKE_BUILD_TYPE supplies -O3 for Release; setting it here as well would
    # silently override a Debug build.
    if(BLACKJACK_NATIVE_ARCH)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
            add_compile_options(-march=native)
        endif()
    endif()

    if(BLACKJACK_SANITIZE)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Game library
set(GAME_SOURCES
    include/game/Card.cpp
    include/game/Deck.cpp
    include/game/Hand.cpp
    include/game/BlackjackGame.cpp
)

add_library(blackjack_game STATIC ${GAME_SOURCES})
target_include_directories(blackjack_game PUBLIC include)
target_compile_features(blackjack_game PUBLIC cxx_std_17)

# MSVC-specific settings
if(MSVC)
    target_compile_options(blackjack_game PRIVATE /W4)
    if(BLACKJACK_WERROR)
        target_compile_options(blackjack_game PRIVATE /WX)
    endif()
endif()

# === AI Library ===
set(AI_SOURCES
    include/ai/State.cpp
    include/ai/PolicyTable.cpp
    include/ai/TabularAgent.cpp
    include/ai/ExpectedSarsaAgent.cpp
    include/ai/DoubleQLearningAgent.cpp
)

add_library(blackjack_ai STATIC ${AI_SOURCES})
target_include_directories(blackjack_ai PUBLIC include)
target_link_libraries(blackjack_ai PUBLIC blackjack_game)

# === Exact solver (depends on AI for Action and State) ===
set(SOLVER_SOURCES
    include/solver/Dealer.cpp
    include/solver/InfiniteDeck.cpp
    include/solver/Solver.cpp
)

add_library(blackjack_solver STATIC ${SOLVER_SOURCES})
target_include_directories(blackjack_solver PUBLIC include)
target_link_libraries(blackjack_solver PUBLIC blackjack_ai)

# === Training (depends on AI) ===
set(TRAINING_SOURCES
    include/training/ConvergenceReport.cpp
    include/training/Evaluator.cpp
    include/training/Experiment.cpp
    include/training/Logger.cpp
    include/training/RunConfig.cpp
    include/training/Trainer.cpp
    include/training/StrategyChart.cpp
)

# === Browser build (emcmake cmake ...) ===
# Only what the web demo plays with: the engine, the agents and the basic
# strategy chart (Evaluator.cpp). No threads, no tests, no CLI tools. Built by
# web/build.sh and the Pages workflow; produces blackjack.mjs + blackjack.wasm.
if(EMSCRIPTEN)
    add_executable(blackjack_wasm bindings/wasm/blackjack_wasm.cpp
                   ${GAME_SOURCES} ${AI_SOURCES} include/training/Evaluator.cpp)
    target_include_directories(blackjack_wasm PRIVATE include)
    # Model loading reports a missing or corrupt file by throwing.
    target_compile_options(blackjack_wasm PRIVATE -fexceptions)
    target_link_options(blackjack_wasm PRIVATE
        -fexceptions -lembind
        -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createBlackjack
        -sENVIRONMENT=web,node -sALLOW_MEMORY_GROWTH=1 -sFORCE_FILESYSTEM=1
        "-sEXPORTED_RUNTIME_METHODS=['FS']")
    set_target_properties(blackjack_wasm PROPERTIES OUTPUT_NAME blackjack SUFFIX ".mjs")
    return()
endif()

add_library(blackjack_training STATIC ${TRAINING_SOURCES})
target_include_directories(blackjack_training PUBLIC include)
target_link_libraries(blackjack_training PUBLIC blackjack_ai blackjack_solver Threads::Threads)

# Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test executable
set(TEST_SOURCES
    tests/test_hand.cpp
    tests/test_game.cpp
    tests/test_q_learning.cpp
    tests/test_trainer.cpp
    tests/test_evaluator.cpp
    tests/test_argparser.cpp
    tests/test_lab_bench.cpp
    tests/test_agents.cpp
    tests/test_counting.cpp
    tests/test_solver.cpp
)

add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests
    blackjack_ai
    blackjack_training
    blackjack_game
    gtest
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_tests)

# === Training Executable ===
add_executable(train scripts/train.cpp)
target_link_libraries(train blackjack_training) 

# === Benchmark Executable ===
add_executable(benchmark scripts/benchmark.cpp)
target_link_libraries(benchmark blackjack_training)

# === Play Executable ===
add_executable(play scripts/play.cpp)
target_link_libraries(play blackjack_training)

# === Experiment Executable (one config over N seeds) ===
add_executable(experiment scripts/experiment.cpp)
target_link_libraries(experiment blackjack_training)

# Create necessary directories
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/checkpoints)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/logs)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/models)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/analysis)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/../config)

# === Build Summary ===
message(STATUS "=================================")
message(STATUS "Blackjack AI - Build Configuration")
message(STATUS "=================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Warnings as errors: ${BLACKJACK_WERROR}")
message(STATUS "Native arch (-march=native): ${BLACKJACK_NATIVE_ARCH}")
message(STATUS "Sanitizers: ${BLACKJACK_SANITIZE}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  - blackjack_game (library)")
message(STATUS "  - blackjack_ai (library)")
message(STATUS "  - train (executable)")
message(STATUS "  - run_tests (executable)")
message(STATUS "  - benchmark (executable)")
message(STATUS "  - play (executable)")
message(STATUS "  - experiment (executable)")
message(STATUS "=================================")