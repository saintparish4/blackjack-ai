#include "Hand.hpp"
#include <sstream>
#include <stdexcept>


namespace blackjack {

void Hand::addCard(const Card &card) { cards_.push_back(card); }

void Hand::clear() { cards_.clear(); }

Hand::Value Hand::getValue() const {
  int total = 0;
  int aces = 0;

  // First pass: count aces as 11
  for (const auto &card : cards_) {
    if (card.isAce()) {
      aces++;
      total += 11;
    } else {
      total += card.getValue();
    }
  }

  // Second pass: convert aces from 11 to 1 if bust
  while (total > 21 && aces > 0) {
    total -= 10; // Convert one ace from 11 to 1
    aces--;
  }

  // Hand is soft if it has at least one ace still counting as 11
  bool isSoft = (aces > 0) && (total <= 21);

  return {total, isSoft};
}

bool Hand::isBlackjack() const {
  if (cards_.size() != 2) {
    return false;
  }

  return getTotal() == 21;
}

bool Hand::canSplit() const {
  if (cards_.size() != 2) {
    return false;
  }

  // Can split if both cards have same rank
  return cards_[0].getRank() == cards_[1].getRank();
}

std::string Hand::toString() const {
  if (cards_.empty()) {
    return "Empty hand";
  }

  std::ostringstream oss;
  oss << "[";

  for (size_t i = 0; i < cards_.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << cards_[i].toShortString();
  }

  oss << "]";

  auto value = getValue();
  oss << " = " << value.total;

  if (value.isSoft) {
    oss << " (soft)";
  }

  if (isBlackjack()) {
    oss << " BLACKJACK!";
  } else if (isBust()) {
    oss << " BUST";
  }

  return oss.str();
}

Card Hand::split() {
  if (!canSplit()) {
    throw std::logic_error("Hand cannot be split");
  }

  Card secondCard = cards_[1];
  cards_.pop_back();

  return secondCard;
}

} // namespace blackjack