#pragma once

#include "Card.hpp"
#include <optional>
#include <string>
#include <vector>

namespace blackjack {

/** Player or dealer hand; value calculation handles soft/hard aces. */
class Hand {
public:
  struct Value {
    int total;
    bool isSoft; // usable ace counting as 11

    bool operator==(const Value &other) const {
      return total == other.total && isSoft == other.isSoft;
    }
  };

  Hand() = default;
  void addCard(const Card &card);
  void clear();

  /** Soft aces count as 11 until that would bust, then as 1. */
  Value getValue() const;

  int getTotal() const { return getValue().total; }
  bool isSoft() const { return getValue().isSoft; }
  bool isBlackjack() const;
  bool isBust() const { return getTotal() > 21; }
  bool canSplit() const;
  size_t size() const { return cards_.size(); }
  bool empty() const { return cards_.empty(); }
  const std::vector<Card> &getCards() const { return cards_; }
  std::string toString() const;

  /** @return Second card (first remains in this hand). @throws std::logic_error
   * if not splittable. */
  Card split();

private:
  std::vector<Card> cards_;
  mutable std::optional<Value> cachedValue_;
};

} // namespace blackjack