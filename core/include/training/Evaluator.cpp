#include "Evaluator.hpp"
#include "../ai/GameStateConverter.hpp"
#include "../util/Seed.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace blackjack {
namespace training {
// === BasicStrategy Implementation ===
//
// Multi-deck, total-dependent basic strategy for the rules this engine plays:
// dealer peeks for blackjack, one split, split aces take one card, and no
// double after split. S17 and H17 differ in a handful of cells, taken from
// rules.dealerHitsSoft17. Dealer upcards are keyed 2-11 (ace = 11).
//
// An earlier, simplified chart had no soft doubles and split pairs as if
// double-after-split were allowed. That weakened the book baseline every EV
// figure is measured against, and marked an agent wrong for doubling soft 17
// against a 4.

BasicStrategy::BasicStrategy(const GameRules &rules) {
  const bool h17 = rules.dealerHitsSoft17;
  initializeHardStrategy(h17);
  initializeSoftStrategy(h17);
  initializePairStrategy();
}

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
  // Without double after split the small pairs lose their weakest splits:
  // 6,6 vs 2, 3,3 and 2,2 vs 2-3, and 4,4 entirely.
  splitVs(6, {3, 4, 5, 6});
  splitVs(3, {4, 5, 6, 7});
  splitVs(2, {4, 5, 6, 7});
  // 5,5 is played as a hard 10 (double), 4,4 as a hard 8, 10,10 stands.
}

void BasicStrategy::initializeHardStrategy(bool h17) {
  for (int dealer = 2; dealer <= 11; ++dealer) {
    for (int player = 4; player <= 21; ++player) {
      ai::Action a;
      if (player <= 8) {
        a = ai::Action::HIT;
      } else if (player == 9) {
        a = (dealer >= 3 && dealer <= 6) ? ai::Action::DOUBLE : ai::Action::HIT;
      } else if (player == 10) {
        a = (dealer <= 9) ? ai::Action::DOUBLE : ai::Action::HIT;
      } else if (player == 11) {
        // Against an ace, doubling 11 only pays when the dealer hits soft 17.
        a = (dealer <= 10 || h17) ? ai::Action::DOUBLE : ai::Action::HIT;
      } else if (player == 12) {
        a = (dealer >= 4 && dealer <= 6) ? ai::Action::STAND : ai::Action::HIT;
      } else if (player <= 16) {
        a = (dealer <= 6) ? ai::Action::STAND : ai::Action::HIT;
      } else {
        a = ai::Action::STAND;
      }
      hardStrategy_[{player, dealer}] = a;
    }
  }

  // Late surrender, applied on top when the rules offer it.
  surrender_.insert({15, 10});
  surrender_.insert({16, 9});
  surrender_.insert({16, 10});
  surrender_.insert({16, 11});
  if (h17) {
    surrender_.insert({15, 11});
    surrender_.insert({17, 11});
  }
}

void BasicStrategy::initializeSoftStrategy(bool h17) {
  auto doubleVs = [&](int total, int lo, int hi) {
    for (int d = lo; d <= hi; ++d) softStrategy_[{total, d}] = ai::Action::DOUBLE;
  };

  for (int player = 12; player <= 21; ++player) {
    for (int dealer = 2; dealer <= 11; ++dealer) {
      softStrategy_[{player, dealer}] =
          (player <= 17) ? ai::Action::HIT : ai::Action::STAND;
    }
  }

  doubleVs(13, 5, 6); // A,2
  doubleVs(14, 5, 6); // A,3
  doubleVs(15, 4, 6); // A,4
  doubleVs(16, 4, 6); // A,5
  doubleVs(17, 3, 6); // A,6

  // Soft 18: double where it pays, otherwise stand, except against 9, T or
  // ace, where hitting is better.
  for (int dealer = 9; dealer <= 11; ++dealer) {
    softStrategy_[{18, dealer}] = ai::Action::HIT;
  }
  for (int dealer = h17 ? 2 : 3; dealer <= 6; ++dealer) {
    softStrategy_[{18, dealer}] = ai::Action::DOUBLE;
    doubleElseStand_.insert({18, dealer});
  }
  if (h17) {
    softStrategy_[{19, 6}] = ai::Action::DOUBLE;
    doubleElseStand_.insert({19, 6});
  }
}

ai::Action BasicStrategy::getAction(const ai::State &state,
                                    bool allowSurrender) const {
  int dealerCard = state.dealerUpCard;
  if (dealerCard == 1)
    dealerCard = 11; // Convert ace to 11 for lookup

  auto key = std::make_pair(state.playerTotal, dealerCard);

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

  const auto &table = state.hasUsableAce ? softStrategy_ : hardStrategy_;
  auto it = table.find(key);
  if (it == table.end()) {
    return state.playerTotal < 17 ? ai::Action::HIT : ai::Action::STAND;
  }

  // Surrender is a two-card decision (canDouble marks the opening hand).
  // Recommending it on a hand that has already hit would mark every agent
  // wrong on a play the rules never offered it.
  if (allowSurrender && state.canDouble && !state.hasUsableAce &&
      surrender_.count(key) > 0) {
    return ai::Action::SURRENDER;
  }
  return it->second;
}

ai::Action BasicStrategy::fallbackWhenCannotDouble(const ai::State &state) const {
  int dealerCard = state.dealerUpCard == 1 ? 11 : state.dealerUpCard;
  if (state.hasUsableAce &&
      doubleElseStand_.count({state.playerTotal, dealerCard}) > 0) {
    return ai::Action::STAND;
  }
  return ai::Action::HIT;
}

bool BasicStrategy::isCorrectAction(const ai::State &state, ai::Action action,
                                    bool allowSurrender) const {
  ai::Action optimalAction = getAction(state, allowSurrender);

  // A hand that has already hit cannot double, so the book's fallback is the
  // correct play there: HIT for most doubles, STAND for soft 18 against a
  // weak card. On a two-card hand it is simply wrong: doubling 11 is the
  // single highest-EV decision in the game, and accepting HIT there hides the
  // agent's most expensive mistake.
  if (optimalAction == ai::Action::DOUBLE && !state.canDouble) {
    return action == fallbackWhenCannotDouble(state);
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

  if (context == HandContext::Pair) {
    // One opening pair per rank (A = 1, ten-value = 10) and upcard. A,A is
    // the only soft pair; every other pair totals twice its rank.
    for (int rank = 1; rank <= 10; ++rank) {
      for (int dealerCard = 1; dealerCard <= 10; ++dealerCard) {
        const bool aces = (rank == 1);
        ai::State state(aces ? 12 : 2 * rank, dealerCard, aces,
                        /*canSplit=*/true, /*canDouble=*/true);
        AuditState entry;
        entry.state = state;
        entry.validActions = legalActionsFor(state, rules);
        states.push_back(std::move(entry));
      }
    }
    return states;
  }

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
  // Book play unavailable here: doubling needs two cards. The chart's
  // fallback is HIT, or STAND on the "double, else stand" soft cells.
  if (desired == ai::Action::DOUBLE) {
    ai::Action fallback = strategy_.fallbackWhenCannotDouble(state);
    if (legal(fallback)) return fallback;
  }
  if (desired == ai::Action::SURRENDER && legal(ai::Action::HIT)) {
    return ai::Action::HIT;
  }
  return validActions.empty() ? ai::Action::STAND : validActions[0];
}

int HiLoAgent::rampUnits(int trueCountFloor, int maxBet) {
  int units = 1;
  if (trueCountFloor >= 5) units = 8;
  else if (trueCountFloor == 4) units = 6;
  else if (trueCountFloor == 3) units = 4;
  else if (trueCountFloor == 2) units = 2;
  return std::max(1, std::min(units, maxBet));
}

const std::vector<IndexPlay> &illustrious18() {
  using ai::Action;
  static const std::vector<IndexPlay> plays = {
      {16, 10, 0, Action::STAND, Action::HIT},
      {15, 10, 4, Action::STAND, Action::HIT},
      {20, 5, 5, Action::SPLIT, Action::STAND, true},
      {20, 6, 4, Action::SPLIT, Action::STAND, true},
      {10, 10, 4, Action::DOUBLE, Action::HIT},
      {12, 3, 2, Action::STAND, Action::HIT},
      {12, 2, 3, Action::STAND, Action::HIT},
      {11, 1, 1, Action::DOUBLE, Action::HIT},
      {9, 2, 1, Action::DOUBLE, Action::HIT},
      {10, 1, 4, Action::DOUBLE, Action::HIT},
      {9, 7, 3, Action::DOUBLE, Action::HIT},
      {16, 9, 5, Action::STAND, Action::HIT},
      {13, 2, -1, Action::STAND, Action::HIT},
      {12, 4, 0, Action::STAND, Action::HIT},
      {12, 5, -2, Action::STAND, Action::HIT},
      {12, 6, -1, Action::STAND, Action::HIT},
      {13, 3, -2, Action::STAND, Action::HIT},
  };
  return plays;
}

std::optional<ai::Action> HiLoAgent::indexPlay(const ai::State &state, int tc) {
  using ai::Action;
  if (state.hasUsableAce) return std::nullopt; // all are hard hands or T,T
  for (const IndexPlay &ix : illustrious18()) {
    if (ix.total != state.playerTotal || ix.dealer != state.dealerUpCard) continue;
    // A pair of 8s is not a hard 16 for index purposes, and T,T plays only
    // apply to a hand that can split.
    if (ix.pair != state.canSplit) continue;
    Action play = tc >= ix.index ? ix.atOrAbove : ix.below;
    // A double index on a hand that has already hit falls back to hitting.
    if (play == Action::DOUBLE && !state.canDouble) play = Action::HIT;
    return play;
  }
  return std::nullopt;
}

ai::Action HiLoAgent::chooseAction(const ai::State &state,
                                   const std::vector<ai::Action> &validActions,
                                   bool training) {
  if (indexPlays_ && state.countBucket != ai::COUNT_UNTRACKED) {
    auto play = indexPlay(state, ai::trueCountOfBucket(state.countBucket));
    if (play && std::find(validActions.begin(), validActions.end(), *play) !=
                    validActions.end()) {
      return *play;
    }
  }
  return book_.chooseAction(state, validActions, training);
}

int HiLoAgent::chooseBet(const ai::BetContext &ctx, bool) {
  const int bucket = ctx.countBucket != ai::COUNT_UNTRACKED
                         ? ctx.countBucket
                         : ai::countBucketFor(ctx.trueCount);
  return std::max(ctx.minBet, rampUnits(ai::trueCountOfBucket(bucket), ctx.maxBet));
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

Evaluator::Evaluator(const GameRules &rules, std::optional<uint32_t> seed)
    : rules_(rules), basicStrategy_(rules), seed_(seed) {}

std::optional<uint32_t> Evaluator::nextDealSeed() {
  if (!seed_) {
    return std::nullopt;
  }
  return util::deriveSeed(*seed_, evaluationCount_++);
}

EvaluationResult Evaluator::evaluate(ai::Agent *agent, size_t numGames,
                                     bool compareStrategy) {
  EvaluationResult result;
  result.gamesPlayed = numGames;

  BlackjackGame game(rules_, nextDealSeed());
  double totalReward = 0.0;
  double totalRewardSq = 0.0;
  // Betting sums for the per-unit-wagered ratio and its delta-method error.
  double sumBet = 0.0, sumBetSq = 0.0, sumWon = 0.0, sumWonSq = 0.0, sumWonBet = 0.0;

  for (size_t i = 0; i < numGames; ++i) {
    double roundReward = 0.0;
    const ai::BetContext ctx = betContext(game);
    const double bet =
        std::clamp(agent->chooseBet(ctx, false), ctx.minBet, ctx.maxBet);
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
      roundReward += ai::GameStateConverter::outcomeToReward(
          outcome, doubled, rules_.blackjackPayout);
    }
    totalReward += roundReward;
    totalRewardSq += roundReward * roundReward;
    result.handsPlayed += outcomes.size();

    const auto bucket = static_cast<size_t>(ctx.countBucket);
    ++result.roundsByCount[bucket];
    result.rewardByCount[bucket] += roundReward;

    const double won = bet * roundReward;
    sumBet += bet;
    sumBetSq += bet * bet;
    sumWon += won;
    sumWonSq += won * won;
    sumWonBet += won * bet;
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

  const double n = static_cast<double>(numGames);
  if (numGames > 0 && sumBet > 0.0) {
    result.avgBet = sumBet / n;
    result.wonPerRound = sumWon / n;
    const double ratio = sumWon / sumBet;
    result.evPerUnitBet = ratio;
    if (numGames > 1) {
      const double wonVar = (sumWonSq - n * result.wonPerRound * result.wonPerRound) / (n - 1.0);
      result.wonStdDev = wonVar > 0.0 ? std::sqrt(wonVar) : 0.0;
      // Var of W - R*B around zero, divided by mean bet squared.
      const double resid =
          (sumWonSq - 2.0 * ratio * sumWonBet + ratio * ratio * sumBetSq) / (n - 1.0);
      result.evPerUnitStdErr =
          resid > 0.0 ? std::sqrt(resid / n) / result.avgBet : 0.0;
    }
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
        playerHand, dealerHand, game.canSplit(), game.canDoubleDown(),
        ai::GameStateConverter::countBucketFor(*agent, game));
    std::vector<ai::Action> validActions =
        ai::GameStateConverter::getValidActions(
            playerHand, game.canSplit(), game.canDoubleDown(),
            game.canSurrender());

    ai::Action action = agent->chooseAction(state, validActions, false);

    ai::GameStateConverter::executeAction(action, game);
  }

  return game.getOutcomes();
}

BaselineComparison Evaluator::compareAgainstBaselines(ai::Agent *agent,
                                                      size_t rounds,
                                                      uint32_t baselineSeed) {
  BaselineComparison cmp;
  BasicStrategyAgent bookAgent(rules_);
  RandomAgent randomAgent(baselineSeed);

  cmp.agent = evaluate(agent, rounds, /*compareStrategy=*/false);
  cmp.book = evaluate(&bookAgent, rounds, /*compareStrategy=*/false);
  cmp.random = evaluate(&randomAgent, rounds, /*compareStrategy=*/false);

  // 0% = plays like random, 100% = plays like the book. A cell-match score
  // cannot express this, because most of the cells it counts almost never
  // come up in a real shoe.
  const double spread = cmp.book.avgReward - cmp.random.avgReward;
  cmp.gapRecovered = spread > 1e-9
                         ? (cmp.agent.avgReward - cmp.random.avgReward) / spread
                         : std::numeric_limits<double>::quiet_NaN();
  return cmp;
}

ai::BetContext Evaluator::betContext(BlackjackGame &game) const {
  game.reshuffleIfDue();
  ai::BetContext ctx;
  ctx.trueCount = game.getTrueCount();
  ctx.countBucket = ai::countBucketFor(ctx.trueCount);
  ctx.minBet = 1;
  ctx.maxBet = std::max(1, rules_.maxBetUnits);
  return ctx;
}

void Evaluator::calibrateBets(ai::Agent *agent, size_t rounds) {
  BlackjackGame game(rules_, nextDealSeed());
  for (size_t i = 0; i < rounds; ++i) {
    const ai::BetContext ctx = betContext(game);
    std::vector<Outcome> outcomes = playGame(agent, game);
    const std::vector<bool> &doubled = game.getWasDoubledByHand();
    double reward = 0.0;
    for (size_t j = 0; j < outcomes.size(); ++j) {
      reward += ai::GameStateConverter::outcomeToReward(
          outcomes[j], j < doubled.size() && doubled[j], rules_.blackjackPayout);
    }
    agent->learnBet(ctx, reward);
  }
}

double Evaluator::compareWithBasicStrategy(ai::Agent *agent) {
  return auditPolicy(agent).accuracy;
}

PolicyAudit Evaluator::auditPolicy(ai::Agent *agent) {
  PolicyAudit audit;

  for (HandContext context : kAllHandContexts) {
    ContextAudit &slot = context == HandContext::TwoCard   ? audit.twoCard
                         : context == HandContext::MultiCard ? audit.multiCard
                                                             : audit.pairs;

    for (const AuditState &entry : buildAuditStates(rules_, context)) {
      // An action the agent was never offered here still holds the Q-table's
      // initial 0.0, which outranks every genuinely negative alternative. The
      // resulting argmax is an artefact of initialisation, not a decision, so
      // scoring it would measure nothing. States like these are reported as
      // coverage gaps instead.
      const ai::State asked = auditView(*agent, entry.state);
      bool trained = true;
      for (ai::Action action : entry.validActions) {
        if (agent->getVisitCount(asked, action) == 0) {
          trained = false;
          break;
        }
      }

      if (!trained) {
        ++slot.untrained;
        continue;
      }

      ++slot.scored;
      ai::Action chosen = agent->chooseAction(asked, entry.validActions, false);
      if (basicStrategy_.isCorrectAction(entry.state, chosen, rules_.surrender)) {
        ++slot.matches;
      }
    }
  }

  const size_t scored =
      audit.twoCard.scored + audit.multiCard.scored + audit.pairs.scored;
  const size_t matches =
      audit.twoCard.matches + audit.multiCard.matches + audit.pairs.matches;
  audit.accuracy = scored > 0 ? static_cast<double>(matches) / scored : 0.0;
  return audit;
}

} // namespace training
} // namespace blackjack