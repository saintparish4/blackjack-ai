#pragma once

#include "Card.hpp"
#include <random>
#include <vector>

namespace blackjack {

/** Deck of cards; supports multiple 52-card decks and Fisher-Yates shuffle. */
class Deck {
public:
  explicit Deck(size_t numDecks = 1);

  void shuffle();

  /** @throws std::runtime_error if deck is empty */
  Card deal();

  /** penetration: fraction of deck dealt before reshuffling (default 0.75). */
  bool needsReshuffle(double penetration = 0.75) const;

  size_t cardsRemaining() const { return cards_.size() - currentIndex_; }
  size_t totalCards() const { return cards_.size(); }
  void reset();

private:
  std::vector<Card> cards_;
  size_t currentIndex_;
  const size_t numDecks_;
  std::mt19937 rng_;

  void initializeDeck();
};

} // namespace blackjack