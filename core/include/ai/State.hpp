#pragma once

#include <functional>
#include <string>

namespace blackjack {
namespace ai {

/** Discrete RL state: player total (4-21), dealer upcard (1-10, Ace=1), soft hand. */
struct State {
  int playerTotal;
  int dealerUpCard;
  bool hasUsableAce;
  bool canSplit = false;   // optional; not in basic Q-learning
  bool canDouble = false;

  // Default constructor for terminal/uninitialized states
  State() : playerTotal(0), dealerUpCard(0), hasUsableAce(false) {}

  State(int playerTotal, int dealerUpCard, bool hasUsableAce)
      : playerTotal(playerTotal), dealerUpCard(dealerUpCard),
        hasUsableAce(hasUsableAce) {}

  State(int playerTotal, int dealerUpCard, bool hasUsableAce, bool canSplit,
        bool canDouble)
      : playerTotal(playerTotal), dealerUpCard(dealerUpCard),
        hasUsableAce(hasUsableAce), canSplit(canSplit), canDouble(canDouble) {}

  /** Bit-packed for Q-table key: total(5) | upcard(4) | ace(1) | split(1) | double(1). */
  size_t hash() const {
    size_t h = 0;
    h |= (static_cast<size_t>(playerTotal) & 0x1F);
    h |= (static_cast<size_t>(dealerUpCard) & 0x0F) << 5;
    h |= (hasUsableAce ? 1ULL : 0ULL) << 9;
    h |= (canSplit ? 1ULL : 0ULL) << 10;
    h |= (canDouble ? 1ULL : 0ULL) << 11;
    return h;
  }

  bool operator==(const State &other) const {
    return playerTotal == other.playerTotal &&
           dealerUpCard == other.dealerUpCard &&
           hasUsableAce == other.hasUsableAce && canSplit == other.canSplit &&
           canDouble == other.canDouble;
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
