#pragma once

#include <cstdint>
#include <ostream>
#include <string>


namespace blackjack {

enum class Rank : uint8_t {
  ACE = 1,
  TWO = 2,
  THREE = 3,
  FOUR = 4,
  FIVE = 5,
  SIX = 6,
  SEVEN = 7,
  EIGHT = 8,
  NINE = 9,
  TEN = 10,
  JACK = 11,
  QUEEN = 12,
  KING = 13
};

enum class Suit : uint8_t { HEARTS, DIAMONDS, CLUBS, SPADES };

/**
 * @brief Represents a single playing card
 *
 * Immutable value type representing a card with rank and suit.
 * Provides utility methods for value calculation and string representation.
 */
class Card {
public:
  /**
   * @brief Construct a card with given rank and suit
   */
  constexpr Card(Rank rank, Suit suit) noexcept : rank_(rank), suit_(suit) {}

  /**
   * @brief Get the card's rank
   */
  constexpr Rank getRank() const noexcept { return rank_; }

  /**
   * @brief Get the card's suit
   */
  constexpr Suit getSuit() const noexcept { return suit_; }

  /**
   * @brief Get the blackjack value of the card
   *
   * Aces are worth 1 (can be 11, handled by Hand class)
   * Face cards (J, Q, K) are worth 10
   * Number cards are worth their rank value
   */
  constexpr int getValue() const noexcept {
    if (rank_ >= Rank::TEN) {
      return 10;
    }
    return static_cast<int>(rank_);
  }

  /**
   * @brief Check if card is an Ace
   */
  constexpr bool isAce() const noexcept { return rank_ == Rank::ACE; }

  /**
   * @brief Get string representation of rank
   */
  std::string getRankString() const;

  /**
   * @brief Get string representation of suit
   */
  std::string getSuitString() const;

  /**
   * @brief Get full string representation (e.g., "Ace of Spades")
   */
  std::string toString() const;

  /**
   * @brief Get short string representation (e.g., "AS")
   */
  std::string toShortString() const;

  // Comparison operators
  constexpr bool operator==(const Card &other) const noexcept {
    return rank_ == other.rank_ && suit_ == other.suit_;
  }

  constexpr bool operator!=(const Card &other) const noexcept {
    return !(*this == other);
  }

  // Stream output
  friend std::ostream &operator<<(std::ostream &os, const Card &card);

private:
  Rank rank_;
  Suit suit_;
};

} // namespace blackjack