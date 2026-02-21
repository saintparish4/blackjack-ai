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
  /** allowSplit/allowDouble: use game.canSplit() and game.canDoubleDown() when
   *  calling from Trainer/Evaluator so split hands get canSplit=false,
   *  canDouble=false (no double-after-split). */
  static State toAIState(const Hand &playerHand, const Hand &dealerHand,
                          bool allowSplit = true, bool allowDouble = true) {
    auto playerValue = playerHand.getValue();

    const auto &dealerCards = dealerHand.getCards();
    if (dealerCards.empty()) {
      throw std::logic_error("Dealer has no cards");
    }

    int dealerUpCard = dealerCards[0].getValue();
    if (dealerCards[0].isAce()) {
      dealerUpCard = 1; // state uses 1 for ace
    }

    bool canSplit = allowSplit && playerHand.canSplit();
    bool canDouble = allowDouble && (playerHand.size() == 2);
    return State(playerValue.total, dealerUpCard, playerValue.isSoft, canSplit,
                 canDouble);
  }

  static std::vector<Action> getValidActions(const Hand &playerHand,
                                             bool allowSplit = true,
                                             bool allowDouble = true,
                                             bool allowSurrender = false) {
    std::vector<Action> actions;
    actions.reserve(5);
    actions.push_back(Action::HIT);
    actions.push_back(Action::STAND);
    if (allowDouble && playerHand.size() == 2) {
      actions.push_back(Action::DOUBLE);
    }
    if (allowSplit && playerHand.canSplit()) {
      actions.push_back(Action::SPLIT);
    }
    if (allowSurrender && playerHand.size() == 2) {
      actions.push_back(Action::SURRENDER);
    }
    return actions;
  }

  /** Executes action in game. */
  static bool executeAction(Action action, BlackjackGame &game) {
    switch (action) {
    case Action::HIT:
      return game.hit();
    case Action::STAND:
      game.stand();
      return true;
    case Action::DOUBLE:
      if (!game.doubleDown()) {
        return game.hit(); // fallback when double not allowed
      }
      return true;
    case Action::SPLIT:
      return game.split();
    case Action::SURRENDER:
      return game.surrender();
    }
    return false;
  }

  /** Rewards: blackjack +1.5, win +1, push 0, loss/bust -1. If wasDoubled, multiply by 2. */
  static double outcomeToReward(Outcome outcome, bool wasDoubled = false) {
    double r;
    switch (outcome) {
    case Outcome::PLAYER_BLACKJACK:
      r = 1.5;
      break;
    case Outcome::PLAYER_WIN:
    case Outcome::DEALER_BUST:
      r = 1.0;
      break;
    case Outcome::PUSH:
      r = 0.0;
      break;
    case Outcome::DEALER_WIN:
    case Outcome::PLAYER_BUST:
      r = -1.0;
      break;
    case Outcome::SURRENDER:
      r = -0.5;
      break;
    default:
      r = 0.0;
    }
    return wasDoubled ? (r * 2.0) : r;
  }
};
} // namespace ai
} // namespace blackjack