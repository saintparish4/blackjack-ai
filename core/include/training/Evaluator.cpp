#include "Evaluator.hpp"
#include "../ai/GameStateConverter.hpp"
#include <random>

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

  // 13-16: Stand vs 2-6, hit vs 7-A
  for (int player = 13; player <= 16; ++player) {
    for (int dealer = 2; dealer <= 11; ++dealer) {
      if (dealer >= 2 && dealer <= 6) {
        hardStrategy_[{player, dealer}] = ai::Action::STAND;
      } else {
        hardStrategy_[{player, dealer}] = ai::Action::HIT;
      }
    }
  }

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
    Outcome outcome = playGame(agent, game);

    // Update counts
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
    }

    totalReward += ai::GameStateConverter::outcomeToReward(outcome);
  }

  // Calculate rates
  result.winRate = static_cast<double>(result.wins) / numGames;
  result.lossRate = static_cast<double>(result.losses) / numGames;
  result.pushRate = static_cast<double>(result.pushes) / numGames;
  result.avgReward = totalReward / numGames;
  result.bustRate = static_cast<double>(result.busts) / numGames;

  // Compare with basic strategy
  if (compareStrategy) {
    result.strategyAccuracy = compareWithBasicStrategy(agent, 1000);
  }

  return result;
}

Outcome Evaluator::playGame(ai::Agent *agent, BlackjackGame &game) {
  game.startRound();

  // Check for immediate blackjack
  if (game.isRoundComplete()) {
    return game.getOutcome();
  }

  // Play agent's turn (no training, pure exploitation)
  while (!game.isRoundComplete()) {
    const Hand &playerHand = game.getPlayerHand();
    const Hand &dealerHand = game.getDealerHand(true);

    ai::State state = ai::GameStateConverter::toAIState(playerHand, dealerHand);
    std::vector<ai::Action> validActions =
        ai::GameStateConverter::getValidActions(playerHand);

    ai::Action action = agent->chooseAction(state, validActions, false);

    // Execute action
    switch (action) {
    case ai::Action::HIT:
      game.hit();
      break;
    case ai::Action::STAND:
      game.stand();
      break;
    case ai::Action::DOUBLE:
      if (!game.doubleDown()) {
        game.hit();
      }
      break;
    case ai::Action::SPLIT:
      // Fallback to hit
      game.hit();
      break;
    }
  }

  return game.getOutcome();
}

double Evaluator::compareWithBasicStrategy(ai::Agent *agent,
                                           size_t numSamples) {
  size_t matches = 0;
  std::mt19937 rng(std::random_device{}());

  // Sample various game states
  std::uniform_int_distribution<int> playerDist(12, 20);
  std::uniform_int_distribution<int> dealerDist(2, 11);
  std::uniform_int_distribution<int> softDist(0, 1);

  for (size_t i = 0; i < numSamples; ++i) {
    int playerTotal = playerDist(rng);
    int dealerCard = dealerDist(rng);
    bool hasUsableAce = (softDist(rng) == 1);

    // Convert dealer 11 to 1 (ace)
    if (dealerCard == 11)
      dealerCard = 1;

    ai::State state(playerTotal, dealerCard, hasUsableAce);

    // Get valid actions (simplified - assume HIT and STAND always valid)
    std::vector<ai::Action> validActions = {ai::Action::HIT, ai::Action::STAND};

    // Add DOUBLE for certain totals
    if (playerTotal >= 9 && playerTotal <= 11) {
      validActions.push_back(ai::Action::DOUBLE);
    }

    // Get agent's action
    ai::Action agentAction = agent->chooseAction(state, validActions, false);

    // Check if it matches basic strategy
    if (basicStrategy_.isCorrectAction(state, agentAction)) {
      matches++;
    }
  }

  return static_cast<double>(matches) / numSamples;
}
} // namespace training
} // namespace blackjack