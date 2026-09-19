#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

namespace blackjack {
namespace ai {

/// countBucket value for agents that don't track the count. Every
/// non-counting state keeps the hash it had before counting existed.
constexpr int COUNT_UNTRACKED = 0;
/// Floor of the Hi-Lo true count, clamped to this range, maps to buckets
/// 1..10. Wider buckets would spread a counter's experience thinner; this
/// range still covers every Illustrious 18 index and the whole bet ramp.
constexpr int MIN_TRUE_COUNT = -3;
constexpr int MAX_TRUE_COUNT = 6;

/** Bucket for a Hi-Lo true count: floor(tc) clamped to [-3, +6], as 1..10. */
inline int countBucketFor(double trueCount) {
  const int tc = static_cast<int>(std::floor(trueCount));
  return std::clamp(tc, MIN_TRUE_COUNT, MAX_TRUE_COUNT) - MIN_TRUE_COUNT + 1;
}

/** Floor of the true count a bucket stands for (the clamp ends mean "or beyond"). */
inline int trueCountOfBucket(int bucket) { return bucket + MIN_TRUE_COUNT - 1; }

/// The bucket for a true count of zero, where counting changes nothing.
/// Audits of a counting agent read its policy here.
constexpr int NEUTRAL_COUNT_BUCKET = 0 - MIN_TRUE_COUNT + 1;

/** Discrete RL state: player total (4-21), dealer upcard (1-10, Ace=1), soft
 *  hand, what's legal, and (for counting agents) the true-count bucket. */
struct State {
  int playerTotal;
  int dealerUpCard;
  bool hasUsableAce;
  bool canSplit = false;   // optional; not in basic Q-learning
  bool canDouble = false;
  int countBucket = COUNT_UNTRACKED; ///< 0, or 1..10 from countBucketFor()

  // Default constructor for terminal/uninitialized states
  State() : playerTotal(0), dealerUpCard(0), hasUsableAce(false) {}

  State(int playerTotal, int dealerUpCard, bool hasUsableAce)
      : playerTotal(playerTotal), dealerUpCard(dealerUpCard),
        hasUsableAce(hasUsableAce) {}

  State(int playerTotal, int dealerUpCard, bool hasUsableAce, bool canSplit,
        bool canDouble)
      : playerTotal(playerTotal), dealerUpCard(dealerUpCard),
        hasUsableAce(hasUsableAce), canSplit(canSplit), canDouble(canDouble) {}

  /** Bit-packed for Q-table key:
   *  total(5) | upcard(4) | ace(1) | split(1) | double(1) | countBucket(4). */
  size_t hash() const {
    size_t h = 0;
    h |= (static_cast<size_t>(playerTotal) & 0x1F);
    h |= (static_cast<size_t>(dealerUpCard) & 0x0F) << 5;
    h |= (hasUsableAce ? 1ULL : 0ULL) << 9;
    h |= (canSplit ? 1ULL : 0ULL) << 10;
    h |= (canDouble ? 1ULL : 0ULL) << 11;
    h |= (static_cast<size_t>(countBucket) & 0x0F) << 12;
    return h;
  }

  bool operator==(const State &other) const {
    return playerTotal == other.playerTotal &&
           dealerUpCard == other.dealerUpCard &&
           hasUsableAce == other.hasUsableAce && canSplit == other.canSplit &&
           canDouble == other.canDouble && countBucket == other.countBucket;
  }

  bool operator!=(const State &other) const { return !(*this == other); }

  bool isValid() const {
    return playerTotal >= 4 && playerTotal <= 21 && dealerUpCard >= 1 &&
           dealerUpCard <= 10;
  }

  std::string toString() const;
};

} // namespace ai
} // namespace blackjack

namespace std {
template <> struct hash<blackjack::ai::State> {
  size_t operator()(const blackjack::ai::State &state) const {
    return state.hash();
  }
};
} // namespace std
