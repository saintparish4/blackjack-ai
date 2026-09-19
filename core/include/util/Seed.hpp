#pragma once

#include <cstdint>

namespace blackjack {
namespace util {

/** Independent RNG streams carved out of one user-facing seed. */
enum class SeedStream : uint32_t {
  AGENT = 1,      ///< Exploration in the learning agent
  TRAIN_DEAL = 2, ///< The shoe the agent trains on
  EVAL_DEAL = 3,  ///< Shoes used by periodic and final evaluations
  BASELINE = 4,   ///< Baseline agents that need randomness (RandomAgent)
  FINAL_EVAL = 5, ///< Shoes for an experiment's end-of-run EV comparison
  BET_CALIBRATION = 6, ///< Shoes a bet-learning agent measures its edge on
};

/**
 * Derive a well-mixed 32-bit seed for one stream from a base seed.
 *
 * One --seed has to drive several generators. Handing each the same value, or
 * seed+1, seed+2, would start mt19937 instances from correlated states; the
 * agent's exploration draws and the shoe's shuffle would then move together.
 * splitmix64 decorrelates them cheaply.
 */
inline uint32_t deriveSeed(uint32_t base, uint32_t stream) {
  uint64_t z = (static_cast<uint64_t>(base) << 32) ^ stream;
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return static_cast<uint32_t>(z ^ (z >> 32));
}

inline uint32_t deriveSeed(uint32_t base, SeedStream stream) {
  return deriveSeed(base, static_cast<uint32_t>(stream));
}

} // namespace util
} // namespace blackjack
