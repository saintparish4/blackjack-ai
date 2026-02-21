#include "Evaluator.hpp"
#include "../ai/GameStateConverter.hpp"

namespace blackjack {
namespace training {
// === BasicStrategy Implementation ===

BasicStrategy::BasicStrategy() {
  initializeHardStrategy();
  initializeSoftStrategy();
}

void BasicStrategy::initializeHardStrategy() {
  // Hard totals strategy (player total, dealer up card) -> Action
  // Based on mathematically optimal basic strategy

  // Always hit on 8 or less
  for (int dealer = 2; dealer <= 11; ++dealer) {
    for (int player = 4; player <= 8; ++player) {
      hardStrategy_[{player, dealer}] = ai::Action::HIT;
    }
  }

  // 9: Double vs 3-6, else hit
  for (int dealer = 2; dealer <= 11; ++dealer) {
    if (dealer >= 3 && dealer <= 6) {
      hardStrategy_[{9, dealer}] = ai::Action::DOUBLE;
    } else {
      hardStrategy_[{9, dealer}] = ai::Action::HIT;
    }
  }

  // 10: Double vs 2-9, else hit
  for (int dealer = 2; dealer <= 11; ++dealer) {
    if (dealer >= 2 && dealer <= 9) {
      hardStrategy_[{10, dealer}] = ai::Action::DOUBLE;
    } else {
      hardStrategy_[{10, dealer}] = ai::Action::HIT;
    }
  }

  // 11: Always double
  for (int dealer = 2; dealer <= 11; ++dealer) {
    hardStrategy_[{11, dealer}] = ai::Action::DOUBLE;
  }

  // 12: Hit vs 2-3 and 7-A, stand vs 4-6
  for (int dealer = 2; dealer <= 11; ++dealer) {
    if (dealer >= 4 && dealer <= 6) {
      hardStrategy_[{12, dealer}] = ai::Action::STAND;
    } else {
      hardStrategy_[{12, dealer}] = ai::Action::HIT;
    }
  }

  // 13-16: Stand vs 2-6, hit vs 7-A (surrender overrides below for 15/16)
  for (int player = 13; player <= 16; ++player) {
    for (int dealer = 2; dealer <= 11; ++dealer) {
      if (dealer >= 2 && dealer <= 6) {
        hardStrategy_[{player, dealer}] = ai::Action::STAND;
      } else {
        hardStrategy_[{player, dealer}] = ai::Action::HIT;
      }
    }
  }
  // Surrender: hard 15 vs 10, hard 16 vs 9/10/A
  hardStrategy_[{15, 10}] = ai::Action::SURRENDER;
  hardStrategy_[{16, 9}] = ai::Action::SURRENDER;
  hardStrategy_[{16, 10}] = ai::Action::SURRENDER;
  hardStrategy_[{16, 11}] = ai::Action::SURRENDER;

  // 17+: Always stand
  for (int player = 17; player <= 21; ++player) {
    for (int dealer = 2; dealer <= 11; ++dealer) {
      hardStrategy_[{player, dealer}] = ai::Action::STAND;
    }
  }
}

void BasicStrategy::initializeSoftStrategy() {
  // Soft totals strategy (Ace counted as 11)

  // Soft 13-17: Hit vs 2-6, hit vs 7-A (simplified)
  // In reality, some soft hands double, but we'll use simplified version
  for (int player = 13; player <= 17; ++player) {
    for (int dealer = 2; dealer <= 11; ++dealer) {
      softStrategy_[{player, dealer}] = ai::Action::HIT;
    }
  }

  // Soft 18: Stand vs 2-8, hit vs 9-A
  for (int dealer = 2; dealer <= 11; ++dealer) {
    if (dealer <= 8) {
      softStrategy_[{18, dealer}] = ai::Action::STAND;
    } else {
      softStrategy_[{18, dealer}] = ai::Action::HIT;
    }
  }

  // Soft 19+: Always stand
  for (int player = 19; player <= 21; ++player) {
    for (int dealer = 2; dealer <= 11; ++dealer) {
      softStrategy_[{player, dealer}] = ai::Action::STAND;
    }
  }
}

ai::Action BasicStrategy::getAction(const ai::State &state) const {
  int dealerCard = state.dealerUpCard;
  if (dealerCard == 1)
    dealerCard = 11; // Convert ace to 11 for lookup

  auto key = std::make_pair(state.playerTotal, dealerCard);

  if (state.hasUsableAce) {
    auto it = softStrategy_.find(key);
    if (it != softStrategy_.end()) {
      return it->second;
    }
  } else {
    auto it = hardStrategy_.find(key);
    if (it != hardStrategy_.end()) {
      return it->second;
    }
  }

  // Default: hit if < 17, stand if >= 17
  return (state.playerTotal < 17) ? ai::Action::HIT : ai::Action::STAND;
}

bool BasicStrategy::isCorrectAction(const ai::State &state,
                                    ai::Action action) const {
  ai::Action optimalAction = getAction(state);

  // DOUBLE can be substituted with HIT
  if (optimalAction == ai::Action::DOUBLE && action == ai::Action::HIT) {
    return true;
  }

  return action == optimalAction;
}

// === Evaluator Implementation ===

Evaluator::Evaluator(const GameRules &rules) : rules_(rules) {}

EvaluationResult Evaluator::evaluate(ai::Agent *agent, size_t numGames,
                                     bool compareStrategy) {
  EvaluationResult result;
  result.gamesPlayed = numGames;

  BlackjackGame game(rules_);
  double totalReward = 0.0;

  for (size_t i = 0; i < numGames; ++i) {
    std::vector<Outcome> outcomes = playGame(agent, game);
    const std::vector<bool> &wasDoubled = game.getWasDoubledByHand();

    for (size_t j = 0; j < outcomes.size(); ++j) {
      Outcome outcome = outcomes[j];
      bool doubled = j < wasDoubled.size() && wasDoubled[j];
      switch (outcome) {
      case Outcome::PLAYER_WIN:
      case Outcome::DEALER_BUST:
        result.wins++;
        break;
      case Outcome::PLAYER_BLACKJACK:
        result.wins++;
        result.blackjacks++;
        break;
      case Outcome::DEALER_WIN:
        result.losses++;
        break;
      case Outcome::PLAYER_BUST:
        result.losses++;
        result.busts++;
        break;
      case Outcome::PUSH:
        result.pushes++;
        break;
      case Outcome::SURRENDER:
        result.losses++;
        break;
      }
      totalReward += ai::GameStateConverter::outcomeToReward(outcome, doubled);
    }
  }

  // Calculate rates
  result.winRate = static_cast<double>(result.wins) / numGames;
  result.lossRate = static_cast<double>(result.losses) / numGames;
  result.pushRate = static_cast<double>(result.pushes) / numGames;
  result.avgReward = totalReward / numGames;
  result.bustRate = static_cast<double>(result.busts) / numGames;

  // Compare with basic strategy
  if (compareStrategy) {
    result.strategyAccuracy = compareWithBasicStrategy(agent);
  }

  return result;
}

std::vector<Outcome> Evaluator::playGame(ai::Agent *agent, BlackjackGame &game) {
  game.startRound();

  if (game.isRoundComplete()) {
    return game.getOutcomes();
  }

  while (!game.isRoundComplete()) {
    const Hand &playerHand = game.getPlayerHand();
    const Hand &dealerHand = game.getDealerHand(true);

    ai::State state = ai::GameStateConverter::toAIState(
        playerHand, dealerHand, game.canSplit(), game.canDoubleDown());
    std::vector<ai::Action> validActions =
        ai::GameStateConverter::getValidActions(
            playerHand, game.canSplit(), game.canDoubleDown(),
            game.canSurrender());

    ai::Action action = agent->chooseAction(state, validActions, false);

    ai::GameStateConverter::executeAction(action, game);
  }

  return game.getOutcomes();
}

double Evaluator::compareWithBasicStrategy(ai::Agent *agent) {
  size_t matches = 0;
  size_t total = 0;

  for (int playerTotal = 4; playerTotal <= 21; ++playerTotal) {
    for (int dealerCard = 1; dealerCard <= 10; ++dealerCard) {
      for (bool hasUsableAce : {false, true}) {
        ai::State state(playerTotal, dealerCard, hasUsableAce);
        if (!state.isValid()) continue;

        std::vector<ai::Action> validActions = {ai::Action::HIT,
                                                ai::Action::STAND};
        if (playerTotal >= 9 && playerTotal <= 11) {
          validActions.push_back(ai::Action::DOUBLE);
        }
        // Surrender-relevant states (two-card hand): hard 15 vs 10, hard 16 vs 9/10/A
        int dealerForLookup = (dealerCard == 1) ? 11 : dealerCard;
        if (!hasUsableAce &&
            ((playerTotal == 15 && dealerForLookup == 10) ||
             (playerTotal == 16 && (dealerForLookup == 9 || dealerForLookup == 10 ||
                                    dealerForLookup == 11)))) {
          validActions.push_back(ai::Action::SURRENDER);
        }

        ai::Action agentAction = agent->chooseAction(state, validActions, false);
        if (basicStrategy_.isCorrectAction(state, agentAction)) {
          matches++;
        }
        total++;
      }
    }
  }

  return total > 0 ? static_cast<double>(matches) / total : 0.0;
}
} // namespace training
} // namespace blackjack