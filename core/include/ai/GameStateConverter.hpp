#pragma once

#include "../game/BlackjackGame.hpp"
#include "../game/Hand.hpp"
#include "Agent.hpp"
#include "State.hpp"

namespace blackjack {
namespace ai {

/** Maps game (Hands) to AI State and valid actions; outcome → reward scale. */
class GameStateConverter {
public:
  static State toAIState(const Hand &playerHand, const Hand &dealerHand) {
    auto playerValue = playerHand.getValue();

    const auto &dealerCards = dealerHand.getCards();
    if (dealerCards.empty()) {
      throw std::logic_error("Dealer has no cards");
    }

    int dealerUpCard = dealerCards[0].getValue();
    if (dealerCards[0].isAce()) {
      dealerUpCard = 1; // state uses 1 for ace
    }

    return State(playerValue.total, dealerUpCard, playerValue.isSoft,
                 playerHand.canSplit(), playerHand.size() == 2);
  }

  static std::vector<Action> getValidActions(const Hand &playerHand) {
    std::vector<Action> actions;
    actions.push_back(Action::HIT);
    actions.push_back(Action::STAND);
    if (playerHand.size() == 2) {
      actions.push_back(Action::DOUBLE);
    }
    if (playerHand.canSplit()) {
      actions.push_back(Action::SPLIT);
    }
    return actions;
  }

  /** Rewards: blackjack +1.5, win +1, push 0, loss/bust -1. */
  static double outcomeToReward(Outcome outcome) {
    switch (outcome) {
    case Outcome::PLAYER_BLACKJACK:
      return 1.5;
    case Outcome::PLAYER_WIN:
    case Outcome::DEALER_BUST:
      return 1.0;
    case Outcome::PUSH:
      return 0.0;
    case Outcome::DEALER_WIN:
    case Outcome::PLAYER_BUST:
      return -1.0;
    default:
      return 0.0;
    }
  }
};
} // namespace ai
} // namespace blackjack