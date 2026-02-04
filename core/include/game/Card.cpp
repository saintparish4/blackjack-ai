#include "Card.hpp"
#include <stdexcept>

namespace blackjack {

std::string Card::getRankString() const {
  switch (rank_) {
  case Rank::ACE:
    return "Ace";
  case Rank::TWO:
    return "2";
  case Rank::THREE:
    return "3";
  case Rank::FOUR:
    return "4";
  case Rank::FIVE:
    return "5";
  case Rank::SIX:
    return "6";
  case Rank::SEVEN:
    return "7";
  case Rank::EIGHT:
    return "8";
  case Rank::NINE:
    return "9";
  case Rank::TEN:
    return "10";
  case Rank::JACK:
    return "Jack";
  case Rank::QUEEN:
    return "Queen";
  case Rank::KING:
    return "King";
  default:
    throw std::logic_error("Invalid rank");
  }
}

std::string Card::getSuitString() const {
  switch (suit_) {
  case Suit::HEARTS:
    return "Hearts";
  case Suit::DIAMONDS:
    return "Diamonds";
  case Suit::CLUBS:
    return "Clubs";
  case Suit::SPADES:
    return "Spades";
  default:
    throw std::logic_error("Invalid suit");
  }
}

std::string Card::toString() const {
  return getRankString() + " of " + getSuitString();
}

std::string Card::toShortString() const {
  std::string rank;
  switch (rank_) {
  case Rank::ACE:
    rank = "A";
    break;
  case Rank::JACK:
    rank = "J";
    break;
  case Rank::QUEEN:
    rank = "Q";
    break;
  case Rank::KING:
    rank = "K";
    break;
  default:
    rank = std::to_string(static_cast<int>(rank_));
  }

  char suit = '?';
  switch (suit_) {
  case Suit::HEARTS:
    suit = 'H';
    break;
  case Suit::DIAMONDS:
    suit = 'D';
    break;
  case Suit::CLUBS:
    suit = 'C';
    break;
  case Suit::SPADES:
    suit = 'S';
    break;
  }

  return rank + suit;
}

std::ostream &operator<<(std::ostream &os, const Card &card) {
  os << card.toShortString();
  return os;
}

} // namespace blackjack