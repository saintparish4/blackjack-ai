cmake_minimum_required(VERSION 3.15)
project(BlackjackAI VERSION 1.0 LANGUAGES CXX)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Werror -pedantic)
    add_compile_options(-O3)  # Optimization for release
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
)

add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests
    blackjack_game
    gtest
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_tests)

# Benchmark executable (uncomment when scripts/benchmark.cpp is created)
# add_executable(benchmark scripts/benchmark.cpp)
# target_link_libraries(benchmark blackjack_game)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")