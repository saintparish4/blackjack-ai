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

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Werror -pedantic)
    add_compile_options(-O3 -march=native)  # Optimization for release
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
    target_compile_options(blackjack_game PRIVATE /W4 /WX)
endif()

# === AI Library ===
set(AI_SOURCES
    include/ai/State.cpp
    include/ai/PolicyTable.cpp
    include/ai/QLearningAgent.cpp
)

# === Training (depends on AI) ===
set(TRAINING_SOURCES
    include/training/Evaluator.cpp
    include/training/Logger.cpp
    include/training/Trainer.cpp
)

add_library(blackjack_ai STATIC ${AI_SOURCES} ${TRAINING_SOURCES})
target_include_directories(blackjack_ai PUBLIC include)
target_link_libraries(blackjack_ai PUBLIC blackjack_game)

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
)

add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests
    blackjack_ai
    blackjack_game
    gtest
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_tests)

# === Benchmark Executable ===
add_executable(benchmark scripts/benchmark.cpp)
target_link_libraries(benchmark blackjack_ai blackjack_game)

# === Build Summary ===
message(STATUS "=================================")
message(STATUS "Blackjack AI - Build Configuration")
message(STATUS "=================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  - blackjack_game (library)")
message(STATUS "  - blackjack_ai (library)")
message(STATUS "  - run_tests (executable)")
message(STATUS "  - benchmark (executable)")
message(STATUS "=================================")