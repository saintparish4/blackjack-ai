#include "BlackjackGame.hpp"
#include <optional>
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
  case Outcome::SURRENDER:
    return "Surrender";
  default:
    return "Unknown";
  }
}

BlackjackGame::BlackjackGame(const GameRules &rules,
                             std::optional<uint32_t> seed)
    : rules_(rules), deck_(std::make_unique<Deck>(rules.numDecks, seed)),
      playerHands_(1), currentHandIndex_(0), splitUsed_(false),
      roundComplete_(false) {}

void BlackjackGame::startRound() {
  checkAndReshuffle();

  playerHands_.clear();
  playerHands_.emplace_back();
  Hand &single = playerHands_.back();
  single.addCard(deck_->deal());
  single.addCard(deck_->deal());

  dealerHand_.clear();
  dealerHand_.addCard(deck_->deal());
  dealerHand_.addCard(deck_->deal());

  currentHandIndex_ = 0;
  splitUsed_ = false;
  roundComplete_ = false;
  outcome_.reset();
  outcomes_.clear();
  doubledByHand_.assign(1, false);

  if (playerHands_[0].isBlackjack() || dealerHand_.isBlackjack()) {
    roundComplete_ = true;
    outcome_ = determineOutcome(playerHands_[0]);
    outcomes_.push_back(*outcome_);
  }
}

bool BlackjackGame::hit() {
  if (roundComplete_) {
    return false;
  }

  Hand &cur = playerHands_[currentHandIndex_];
  cur.addCard(deck_->deal());

  if (cur.isBust()) {
    if (currentHandIndex_ + 1 < playerHands_.size()) {
      currentHandIndex_++;
    } else {
      finishRoundAndResolveOutcomes();
    }
    return true;
  }

  return true;
}

void BlackjackGame::stand() {
  if (roundComplete_) {
    return;
  }

  if (currentHandIndex_ + 1 < playerHands_.size()) {
    currentHandIndex_++;
  } else {
    finishRoundAndResolveOutcomes();
  }
}

bool BlackjackGame::doubleDown() {
  if (!canDoubleDown()) {
    return false;
  }

  doubledByHand_[currentHandIndex_] = true;
  Hand &cur = playerHands_[currentHandIndex_];
  cur.addCard(deck_->deal());

  if (cur.isBust()) {
    if (currentHandIndex_ + 1 < playerHands_.size()) {
      currentHandIndex_++;
    } else {
      finishRoundAndResolveOutcomes();
    }
  } else {
    if (currentHandIndex_ + 1 < playerHands_.size()) {
      currentHandIndex_++;
    } else {
      finishRoundAndResolveOutcomes();
    }
  }

  return true;
}

bool BlackjackGame::surrender() {
  if (!canSurrender()) {
    return false;
  }
  outcome_ = Outcome::SURRENDER;
  outcomes_.assign(1, Outcome::SURRENDER);
  roundComplete_ = true;
  return true;
}

bool BlackjackGame::split() {
  if (!canSplit()) {
    return false;
  }

  Hand &first = playerHands_[0];
  Card secondCard = first.split();

  Hand secondHand;
  secondHand.addCard(secondCard);
  secondHand.addCard(deck_->deal());
  first.addCard(deck_->deal());

  playerHands_.push_back(std::move(secondHand));
  doubledByHand_.push_back(false);
  splitUsed_ = true;
  currentHandIndex_ = 0;

  return true;
}

Outcome BlackjackGame::getOutcome() const {
  if (!roundComplete_) {
    throw std::logic_error("Round is not complete");
  }
  return outcome_.value();
}

const Hand &BlackjackGame::getPlayerHand() const {
  return playerHands_[currentHandIndex_];
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
  if (roundComplete_) {
    return false;
  }
  const Hand &cur = playerHands_[currentHandIndex_];
  if (cur.size() != 2) {
    return false;
  }
  // No double after split in this implementation
  if (splitUsed_ && playerHands_.size() > 1) {
    return false;
  }
  return true;
}

bool BlackjackGame::canSplit() const {
  if (roundComplete_ || splitUsed_) {
    return false;
  }
  if (playerHands_.size() != 1) {
    return false;
  }
  return playerHands_[0].canSplit();
}

bool BlackjackGame::canSurrender() const {
  if (roundComplete_ || !rules_.surrender) {
    return false;
  }
  if (playerHands_.size() != 1) {
    return false;
  }
  return playerHands_[0].size() == 2;
}

void BlackjackGame::reset() {
  deck_->reset();
  playerHands_.clear();
  playerHands_.emplace_back();
  doubledByHand_.assign(1, false);
  dealerHand_.clear();
  currentHandIndex_ = 0;
  splitUsed_ = false;
  roundComplete_ = false;
  outcome_.reset();
  outcomes_.clear();
}

void BlackjackGame::playDealerHand() {
  while (true) {
    int total = dealerHand_.getTotal();
    bool soft = dealerHand_.isSoft();

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
    if (dealerHand_.isBust()) {
      break;
    }
  }
}

Outcome BlackjackGame::determineOutcome(const Hand &playerHand) const {
  bool playerBlackjack = playerHand.isBlackjack();
  bool dealerBlackjack = dealerHand_.isBlackjack();

  if (playerBlackjack && dealerBlackjack) {
    return Outcome::PUSH;
  }
  if (playerBlackjack) {
    return Outcome::PLAYER_BLACKJACK;
  }
  if (dealerBlackjack) {
    return Outcome::DEALER_WIN;
  }

  int playerTotal = playerHand.getTotal();
  int dealerTotal = dealerHand_.getTotal();

  if (playerTotal > 21) {
    return Outcome::PLAYER_BUST;
  }
  if (dealerTotal > 21) {
    return Outcome::DEALER_BUST;
  }

  if (playerTotal > dealerTotal) {
    return Outcome::PLAYER_WIN;
  }
  if (dealerTotal > playerTotal) {
    return Outcome::DEALER_WIN;
  }
  return Outcome::PUSH;
}

void BlackjackGame::finishRoundAndResolveOutcomes() {
  playDealerHand();
  outcomes_.clear();
  for (const Hand &h : playerHands_) {
    outcomes_.push_back(determineOutcome(h));
  }
  outcome_ = outcomes_[0];
  roundComplete_ = true;
}

void BlackjackGame::checkAndReshuffle() {
  if (deck_->needsReshuffle(rules_.penetration)) {
    deck_->reset();
  }
}

} // namespace blackjack
