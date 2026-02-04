#include "BlackjackGame.hpp"
#include <stdexcept>

namespace blackjack {

std::string outcomeToString(Outcome outcome) {
  switch (outcome) {
  case Outcome::PLAYER_WIN:
    return "Player Win";
  case Outcome::PLAYER_BLACKJACK:
    return "Player Blackjack";
  case Outcome::DEALER_WIN:
    return "Dealer Win";
  case Outcome::PUSH:
    return "Push";
  case Outcome::PLAYER_BUST:
    return "Player Bust";
  case Outcome::DEALER_BUST:
    return "Dealer Bust";
  default:
    return "Unknown";
  }
}

BlackjackGame::BlackjackGame(const GameRules &rules)
    : rules_(rules), deck_(std::make_unique<Deck>(rules.numDecks)),
      roundComplete_(false), handCount_(0) {}

void BlackjackGame::startRound() {
  // Check if we need to reshuffle
  checkAndReshuffle();

  // Clear previous hands
  playerHand_.clear();
  dealerHand_.clear();
  roundComplete_ = false;
  outcome_.reset();
  handCount_ = 0;

  // Deal initial cards (player, dealer, player, dealer)
  playerHand_.addCard(deck_->deal());
  dealerHand_.addCard(deck_->deal());
  playerHand_.addCard(deck_->deal());
  dealerHand_.addCard(deck_->deal());

  handCount_ = 2;

  // Check for immediate blackjack
  if (playerHand_.isBlackjack() || dealerHand_.isBlackjack()) {
    roundComplete_ = true;
    outcome_ = determineOutcome();
  }
}

bool BlackjackGame::hit() {
  if (roundComplete_) {
    return false;
  }

  playerHand_.addCard(deck_->deal());
  handCount_++;

  // Check if player busts
  if (playerHand_.isBust()) {
    roundComplete_ = true;
    outcome_ = Outcome::PLAYER_BUST;
    return true;
  }

  return true;
}

void BlackjackGame::stand() {
  if (roundComplete_) {
    return;
  }

  // Player is done, dealer plays
  playDealerHand();

  roundComplete_ = true;
  outcome_ = determineOutcome();
}

bool BlackjackGame::doubleDown() {
  if (!canDoubleDown()) {
    return false;
  }

  // Take one card and end turn
  playerHand_.addCard(deck_->deal());
  handCount_++;

  if (playerHand_.isBust()) {
    roundComplete_ = true;
    outcome_ = Outcome::PLAYER_BUST;
  } else {
    playDealerHand();
    roundComplete_ = true;
    outcome_ = determineOutcome();
  }

  return true;
}

Outcome BlackjackGame::getOutcome() const {
  if (!roundComplete_) {
    throw std::logic_error("Round is not complete");
  }

  return outcome_.value();
}

Hand BlackjackGame::getDealerHand(bool hideHoleCard) const {
  if (hideHoleCard && dealerHand_.size() >= 2) {
    Hand visibleHand;
    visibleHand.addCard(dealerHand_.getCards()[0]);
    return visibleHand;
  }

  return dealerHand_;
}

bool BlackjackGame::canDoubleDown() const {
  // Can only double on first two cards
  return !roundComplete_ && handCount_ == 2;
}

void BlackjackGame::reset() {
  deck_->reset();
  playerHand_.clear();
  dealerHand_.clear();
  roundComplete_ = false;
  outcome_.reset();
  handCount_ = 0;
}

void BlackjackGame::playDealerHand() {
  // Dealer must hit until 17 or higher
  while (true) {
    int total = dealerHand_.getTotal();
    bool soft = dealerHand_.isSoft();

    // Check if dealer should hit
    bool shouldHit = false;

    if (total < 17) {
      shouldHit = true;
    } else if (total == 17 && soft && rules_.dealerHitsSoft17) {
      shouldHit = true;
    }

    if (!shouldHit) {
      break;
    }

    dealerHand_.addCard(deck_->deal());

    // Check for bust
    if (dealerHand_.isBust()) {
      break;
    }
  }
}

Outcome BlackjackGame::determineOutcome() const {
  bool playerBlackjack = playerHand_.isBlackjack();
  bool dealerBlackjack = dealerHand_.isBlackjack();

  // Check for blackjacks first
  if (playerBlackjack && dealerBlackjack) {
    return Outcome::PUSH;
  }

  if (playerBlackjack) {
    return Outcome::PLAYER_BLACKJACK;
  }

  if (dealerBlackjack) {
    return Outcome::DEALER_WIN;
  }

  // Check for busts
  int playerTotal = playerHand_.getTotal();
  int dealerTotal = dealerHand_.getTotal();

  if (playerTotal > 21) {
    return Outcome::PLAYER_BUST;
  }

  if (dealerTotal > 21) {
    return Outcome::DEALER_BUST;
  }

  // Compare totals
  if (playerTotal > dealerTotal) {
    return Outcome::PLAYER_WIN;
  } else if (dealerTotal > playerTotal) {
    return Outcome::DEALER_WIN;
  } else {
    return Outcome::PUSH;
  }
}

void BlackjackGame::checkAndReshuffle() {
  if (deck_->needsReshuffle(rules_.penetration)) {
    deck_->reset();
  }
}

} // namespace blackjack