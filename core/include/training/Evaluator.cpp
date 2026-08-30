#include "Evaluator.hpp"
#include "../ai/GameStateConverter.hpp"
#include <algorithm>
#include <cmath>

namespace blackjack {
namespace training {
// === BasicStrategy Implementation ===

BasicStrategy::BasicStrategy() {
  initializeHardStrategy();
  initializeSoftStrategy();
  initializePairStrategy();
}

// Pair plays for 6-deck, dealer stands soft 17, double-after-split allowed.
// Keyed by the rank held twice (A = 1, ten-value = 10). Entries not listed
// here are never split; the caller falls back to the hard/soft table.
void BasicStrategy::initializePairStrategy() {
  auto splitVs = [&](int rank, std::initializer_list<int> dealers) {
    for (int d : dealers) {
      pairStrategy_[{rank, d}] = ai::Action::SPLIT;
    }
  };

  splitVs(1, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}); // A,A: always
  splitVs(8, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}); // 8,8: always
  splitVs(9, {2, 3, 4, 5, 6, 8, 9});            // 9,9: not vs 7, 10 or A
  splitVs(7, {2, 3, 4, 5, 6, 7});
  splitVs(6, {2, 3, 4, 5, 6});
  splitVs(4, {5, 6});
  splitVs(3, {2, 3, 4, 5, 6, 7});
  splitVs(2, {2, 3, 4, 5, 6, 7});
  // 5,5 is played as a hard 10 (double), never split.
  // 10,10 stands.
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

ai::Action BasicStrategy::getAction(const ai::State &state,
                                    bool allowSurrender) const {
  int dealerCard = state.dealerUpCard;
  if (dealerCard == 1)
    dealerCard = 11; // Convert ace to 11 for lookup

  auto key = std::make_pair(state.playerTotal, dealerCard);
  ai::Action action;

  if (state.canSplit) {
    // A,A is the only pair that is also soft; every other pair totals 2x its
    // rank. A pair with no SPLIT entry falls through to the hard/soft tables.
    int rank = (state.hasUsableAce && state.playerTotal == 12)
                   ? 1
                   : state.playerTotal / 2;
    auto pairIt = pairStrategy_.find({rank, dealerCard});
    if (pairIt != pairStrategy_.end()) {
      return pairIt->second;
    }
  }

  if (state.hasUsableAce) {
    auto it = softStrategy_.find(key);
    action = (it != softStrategy_.end())
                 ? it->second
                 : ((state.playerTotal < 17) ? ai::Action::HIT
                                             : ai::Action::STAND);
  } else {
    auto it = hardStrategy_.find(key);
    action = (it != hardStrategy_.end())
                 ? it->second
                 : ((state.playerTotal < 17) ? ai::Action::HIT
                                             : ai::Action::STAND);
  }

  // Every state the table marks SURRENDER (hard 15 v 10, hard 16 v 9/10/A) is
  // a hit under rules that do not offer it.
  if (action == ai::Action::SURRENDER && !allowSurrender) {
    return ai::Action::HIT;
  }
  return action;
}

bool BasicStrategy::isCorrectAction(const ai::State &state, ai::Action action,
                                    bool allowSurrender) const {
  ai::Action optimalAction = getAction(state, allowSurrender);

  // A hand that has already hit cannot double, so HIT is the correct play
  // wherever the book says DOUBLE. On a two-card hand it is simply wrong:
  // doubling 11 is the single highest-EV decision in the game, and accepting
  // HIT there hides the agent's most expensive mistake.
  if (optimalAction == ai::Action::DOUBLE && action == ai::Action::HIT &&
      !state.canDouble) {
    return true;
  }

  return action == optimalAction;
}

// === Audit state enumeration ===

std::vector<ai::Action> legalActionsFor(const ai::State &state,
                                       const GameRules &rules) {
  std::vector<ai::Action> valid = {ai::Action::HIT, ai::Action::STAND};

  // BlackjackGame::canDoubleDown() permits any two-card hand, so this must
  // too. Restricting DOUBLE to totals 9-11 means the agent is never scored on
  // the error of doubling a 16.
  if (state.canDouble) {
    valid.push_back(ai::Action::DOUBLE);
  }
  if (state.canSplit) {
    valid.push_back(ai::Action::SPLIT);
  }
  // Late surrender is a two-card-only decision, and only when the house offers
  // it. Under rules that do not, SURRENDER is never trained, so its Q stays at
  // the 0.0 default and beats every real (negative) action on these hands.
  if (rules.surrender && state.canDouble) {
    valid.push_back(ai::Action::SURRENDER);
  }

  return valid;
}

std::vector<AuditState> buildAuditStates(const GameRules &rules,
                                         HandContext context) {
  std::vector<AuditState> states;
  const bool twoCard = (context == HandContext::TwoCard);

  for (int playerTotal = 4; playerTotal <= 21; ++playerTotal) {
    for (int dealerCard = 1; dealerCard <= 10; ++dealerCard) {
      for (bool soft : {false, true}) {
        // canSplit stays false: pair states hash to yet another slot and are
        // scored against the pair table, not the hard/soft chart audited here.
        ai::State state(playerTotal, dealerCard, soft, /*canSplit=*/false,
                        /*canDouble=*/twoCard);
        if (!state.isValid()) continue;

        AuditState entry;
        entry.state = state;
        entry.validActions = legalActionsFor(state, rules);
        states.push_back(std::move(entry));
      }
    }
  }

  return states;
}

// === Baseline agents ===

ai::Action BasicStrategyAgent::chooseAction(
    const ai::State &state, const std::vector<ai::Action> &validActions, bool) {
  ai::Action desired = strategy_.getAction(state, allowSurrender_);

  auto legal = [&](ai::Action a) {
    return std::find(validActions.begin(), validActions.end(), a) !=
           validActions.end();
  };

  if (legal(desired)) {
    return desired;
  }
  // Book play unavailable here: doubling needs two cards, splitting needs a
  // pair. Falling back to HIT matches how the chart is read at the table.
  if (desired == ai::Action::DOUBLE && legal(ai::Action::HIT)) {
    return ai::Action::HIT;
  }
  if (desired == ai::Action::SURRENDER && legal(ai::Action::HIT)) {
    return ai::Action::HIT;
  }
  return validActions.empty() ? ai::Action::STAND : validActions[0];
}

ai::Action RandomAgent::chooseAction(const ai::State &,
                                     const std::vector<ai::Action> &validActions,
                                     bool) {
  if (validActions.empty()) {
    return ai::Action::STAND;
  }
  std::uniform_int_distribution<size_t> dist(0, validActions.size() - 1);
  return validActions[dist(rng_)];
}

// === Evaluator Implementation ===

Evaluator::Evaluator(const GameRules &rules) : rules_(rules) {}

EvaluationResult Evaluator::evaluate(ai::Agent *agent, size_t numGames,
                                     bool compareStrategy) {
  EvaluationResult result;
  result.gamesPlayed = numGames;

  BlackjackGame game(rules_);
  double totalReward = 0.0;
  double totalRewardSq = 0.0;

  for (size_t i = 0; i < numGames; ++i) {
    double roundReward = 0.0;
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
      roundReward += ai::GameStateConverter::outcomeToReward(outcome, doubled);
    }
    totalReward += roundReward;
    totalRewardSq += roundReward * roundReward;
    result.handsPlayed += outcomes.size();
  }

  // Outcome rates are per settled hand — a split round contributes two hands —
  // while reward stays per round, since one round is one initial bet.
  const double hands = result.handsPlayed > 0
                           ? static_cast<double>(result.handsPlayed)
                           : 1.0;
  result.winRate = static_cast<double>(result.wins) / hands;
  result.lossRate = static_cast<double>(result.losses) / hands;
  result.pushRate = static_cast<double>(result.pushes) / hands;
  result.avgReward = totalReward / numGames;
  result.bustRate = static_cast<double>(result.busts) / hands;

  // Per-round reward has variance around 1.15 in blackjack, so two agents a
  // few thousandths apart in mean are indistinguishable without this.
  if (numGames > 1) {
    const double mean = result.avgReward;
    const double variance =
        (totalRewardSq / numGames - mean * mean) *
        (static_cast<double>(numGames) / static_cast<double>(numGames - 1));
    result.rewardStdErr =
        variance > 0.0 ? std::sqrt(variance / static_cast<double>(numGames))
                       : 0.0;
  }

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
  return auditPolicy(agent).accuracy;
}

PolicyAudit Evaluator::auditPolicy(ai::Agent *agent) {
  PolicyAudit audit;

  for (HandContext context : {HandContext::TwoCard, HandContext::MultiCard}) {
    ContextAudit &slot = (context == HandContext::TwoCard) ? audit.twoCard
                                                           : audit.multiCard;

    for (const AuditState &entry : buildAuditStates(rules_, context)) {
      // An action the agent was never offered here still holds the Q-table's
      // initial 0.0, which outranks every genuinely negative alternative. The
      // resulting argmax is an artefact of initialisation, not a decision, so
      // scoring it would measure nothing. States like these are reported as
      // coverage gaps instead.
      bool trained = true;
      for (ai::Action action : entry.validActions) {
        if (agent->getVisitCount(entry.state, action) == 0) {
          trained = false;
          break;
        }
      }

      if (!trained) {
        ++slot.untrained;
        continue;
      }

      ++slot.scored;
      ai::Action chosen =
          agent->chooseAction(entry.state, entry.validActions, false);
      if (basicStrategy_.isCorrectAction(entry.state, chosen, rules_.surrender)) {
        ++slot.matches;
      }
    }
  }

  const size_t scored = audit.twoCard.scored + audit.multiCard.scored;
  const size_t matches = audit.twoCard.matches + audit.multiCard.matches;
  audit.accuracy = scored > 0 ? static_cast<double>(matches) / scored : 0.0;
  return audit;
}

} // namespace training
} // namespace blackjack