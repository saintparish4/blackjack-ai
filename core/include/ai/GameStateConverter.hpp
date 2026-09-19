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
  /** countBucket: countBucketFor(game.getTrueCount()) for agents that
   *  usesCount(), COUNT_UNTRACKED for everyone else. */
  static State toAIState(const Hand &playerHand, const Hand &dealerHand,
                          bool allowSplit = true, bool allowDouble = true,
                          int countBucket = COUNT_UNTRACKED) {
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
    State state(playerValue.total, dealerUpCard, playerValue.isSoft, canSplit,
                canDouble);
    state.countBucket = countBucket;
    return state;
  }

  /** The count bucket an agent should see in this game right now. */
  static int countBucketFor(const Agent &agent, const BlackjackGame &game) {
    return agent.usesCount() ? ai::countBucketFor(game.getTrueCount())
                             : COUNT_UNTRACKED;
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

  /** Rewards: blackjack pays blackjackPayout (+1.5 at 3:2, +1.2 at 6:5), win
   *  +1, push 0, loss/bust -1. If wasDoubled, multiply by 2.
   *
   *  Pass GameRules::blackjackPayout. Hard-coding 1.5 here once meant a 6:5
   *  table trained and scored as if it paid 3:2, overstating every EV on it. */
  static double outcomeToReward(Outcome outcome, bool wasDoubled = false,
                                double blackjackPayout = 1.5) {
    double r;
    switch (outcome) {
    case Outcome::PLAYER_BLACKJACK:
      r = blackjackPayout;
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