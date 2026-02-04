#include "Deck.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

namespace blackjack {

Deck::Deck(size_t numDecks)
    : currentIndex_(0), numDecks_(numDecks), rng_(std::random_device{}()) {
  if (numDecks == 0) {
    throw std::invalid_argument("Number of decks must be at least 1");
  }

  initializeDeck();
  shuffle();
}

void Deck::initializeDeck() {
  cards_.clear();
  cards_.reserve(52 * numDecks_);

  for (size_t deck = 0; deck < numDecks_; ++deck) {
    for (int suit = 0; suit < 4; ++suit) {
      for (int rank = 1; rank <= 13; ++rank) {
        cards_.emplace_back(static_cast<Rank>(rank), static_cast<Suit>(suit));
      }
    }
  }
}

void Deck::shuffle() {
  // Fisher-Yates shuffle
  for (size_t i = cards_.size() - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    size_t j = dist(rng_);
    std::swap(cards_[i], cards_[j]);
  }

  currentIndex_ = 0;
}

Card Deck::deal() {
  if (currentIndex_ >= cards_.size()) {
    throw std::runtime_error("Deck is empty");
  }

  return cards_[currentIndex_++];
}

bool Deck::needsReshuffle(double penetration) const {
  if (penetration < 0.0 || penetration > 1.0) {
    throw std::invalid_argument("Penetration must be between 0 and 1");
  }

  size_t cardsDealt = currentIndex_;
  size_t threshold = static_cast<size_t>(cards_.size() * penetration);

  return cardsDealt >= threshold;
}

void Deck::reset() {
  initializeDeck();
  shuffle();
}

} // namespace blackjack