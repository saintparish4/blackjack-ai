#include "ai/QLearningAgent.hpp"
#include "training/Evaluator.hpp"
#include <gtest/gtest.h>
#include <algorithm>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;

class EvaluatorTest : public ::testing::Test {
protected:
  Evaluator evaluator;
  std::shared_ptr<QLearningAgent> agent;

  void SetUp() override {
    QLearningAgent::Hyperparameters params;
    params.epsilon = 0.0;
    params.epsilonMin = 0.0;
    agent = std::make_shared<QLearningAgent>(params);
  }
};

namespace {

/// Splits whenever the rules allow it, so evaluation rounds settle more hands
/// than they deal rounds. Everything else stands.
class AlwaysSplitAgent : public Agent {
public:
  Action chooseAction(const State &, const std::vector<Action> &validActions,
                      bool) override {
    for (Action a : validActions) {
      if (a == Action::SPLIT) {
        return Action::SPLIT;
      }
    }
    return Action::STAND;
  }

  void learn(const Experience &) override {}
  double getQValue(const State &, Action) const override { return 0.0; }
  void save(const std::string &) const override {}
  void load(const std::string &) override {}
  std::string getName() const override { return "AlwaysSplit"; }
};

} // namespace

// === Metrics correctness ===

TEST_F(EvaluatorTest, WinLossPushCountsSumToHandsPlayed) {
  auto result = evaluator.evaluate(agent.get(), 100, false);

  EXPECT_EQ(result.gamesPlayed, 100u);
  EXPECT_EQ(result.wins + result.losses + result.pushes, result.handsPlayed);
}

TEST_F(EvaluatorTest, RatesMatchCounts) {
  auto result = evaluator.evaluate(agent.get(), 200, false);

  EXPECT_NEAR(result.winRate,
              static_cast<double>(result.wins) / result.handsPlayed, 1e-9);
  EXPECT_NEAR(result.lossRate,
              static_cast<double>(result.losses) / result.handsPlayed, 1e-9);
  EXPECT_NEAR(result.pushRate,
              static_cast<double>(result.pushes) / result.handsPlayed, 1e-9);
}

// A split round settles two hands off one deal. Rates are per hand, so they
// must still sum to 1.0 even when handsPlayed exceeds gamesPlayed — dividing
// hand counts by gamesPlayed pushed this sum above 1.0.
TEST_F(EvaluatorTest, RatesSumToOneWhenHandsAreSplit) {
  AlwaysSplitAgent splitter;
  auto result = evaluator.evaluate(&splitter, 500, false);

  ASSERT_GT(result.handsPlayed, result.gamesPlayed)
      << "expected at least one split across 500 rounds";
  EXPECT_EQ(result.wins + result.losses + result.pushes, result.handsPlayed);
  EXPECT_NEAR(result.winRate + result.lossRate + result.pushRate, 1.0, 1e-9);
}

TEST_F(EvaluatorTest, HandsPlayedIsAtLeastGamesPlayed) {
  auto result = evaluator.evaluate(agent.get(), 100, false);

  EXPECT_GE(result.handsPlayed, result.gamesPlayed);
}

// === Surrender availability in strategy comparison ===

// Under rules without surrender the action is never trained, so its Q-value
// stays at the 0.0 initial value. Hard 15/16 against a strong upcard are losing
// hands whose trained actions all carry negative Q, so offering SURRENDER here
// lets an untouched 0.0 win the argmax — the agent then "matches" the book's
// surrender recommendation on four states it never learned.
TEST_F(EvaluatorTest, SurrenderIsNotOfferedWhenRulesDisallowIt) {
  GameRules noSurrender = GameRules::vegasStrip();
  ASSERT_FALSE(noSurrender.surrender);
  Evaluator strict(noSurrender);

  // A fresh agent has every Q-value at 0.0 — the exact condition that made
  // spurious surrenders win.
  QLearningAgent::Hyperparameters params;
  params.epsilon = 0.0;
  params.epsilonMin = 0.0;
  QLearningAgent untrained(params);

  for (int total : {15, 16}) {
    for (int dealer : {9, 10, 1}) {
      if (total == 15 && dealer != 10) continue;
      State state(total, dealer, false);
      auto valid = std::vector<Action>{Action::HIT, Action::STAND};
      EXPECT_NE(untrained.chooseAction(state, valid, false), Action::SURRENDER);
    }
  }

  // Book play for these states collapses to HIT without surrender available.
  const BasicStrategy &book = strict.getBasicStrategy();
  EXPECT_EQ(book.getAction(State(16, 10, false), /*allowSurrender=*/false),
            Action::HIT);
  EXPECT_EQ(book.getAction(State(15, 10, false), /*allowSurrender=*/false),
            Action::HIT);
  // ...and remains SURRENDER when the rules do offer it.
  EXPECT_EQ(book.getAction(State(16, 10, false), /*allowSurrender=*/true),
            Action::SURRENDER);
}

TEST_F(EvaluatorTest, SurrenderRulesChangeTheOfferedActionSet) {
  GameRules withSurrender = GameRules::atlanticCity();
  GameRules withoutSurrender = GameRules::vegasStrip();
  ASSERT_TRUE(withSurrender.surrender);
  ASSERT_FALSE(withoutSurrender.surrender);

  State twoCard(16, 10, false, /*canSplit=*/false, /*canDouble=*/true);

  auto offered = legalActionsFor(twoCard, withSurrender);
  auto notOffered = legalActionsFor(twoCard, withoutSurrender);

  // Under rules that do not offer surrender the action is never trained, so
  // its Q stays at the 0.0 default and would beat every real (negative) action
  // on a hand this bad. It must not be in the action set at all.
  EXPECT_NE(std::find(offered.begin(), offered.end(), Action::SURRENDER),
            offered.end());
  EXPECT_EQ(std::find(notOffered.begin(), notOffered.end(), Action::SURRENDER),
            notOffered.end());
}

TEST_F(EvaluatorTest, UntrainedAgentIsReportedAsUnscoredNotAsAccurate) {
  QLearningAgent::Hyperparameters params;
  params.epsilon = 0.0;
  params.epsilonMin = 0.0;
  QLearningAgent untrained(params);

  PolicyAudit audit = evaluator.auditPolicy(&untrained);

  // Every Q-value is still the 0.0 default, so any argmax just reports action
  // enum order. Scoring that would manufacture an accuracy number out of
  // nothing, which is how a never-trained DOUBLE used to pass as a decision.
  EXPECT_EQ(audit.twoCard.scored, 0u);
  EXPECT_EQ(audit.multiCard.scored, 0u);
  EXPECT_GT(audit.twoCard.untrained, 0u);
  EXPECT_GT(audit.multiCard.untrained, 0u);
  EXPECT_DOUBLE_EQ(audit.accuracy, 0.0);
}

TEST_F(EvaluatorTest, AuditCoversTwoCardAndMultiCardPoliciesSeparately) {
  GameRules rules = GameRules::vegasStrip();

  auto twoCard = buildAuditStates(rules, HandContext::TwoCard);
  auto multiCard = buildAuditStates(rules, HandContext::MultiCard);

  ASSERT_FALSE(twoCard.empty());
  ASSERT_EQ(twoCard.size(), multiCard.size());

  for (size_t i = 0; i < twoCard.size(); ++i) {
    // canDouble is packed into State::hash(), so the two contexts address
    // different Q-table slots. Auditing one says nothing about the other.
    EXPECT_TRUE(twoCard[i].state.canDouble);
    EXPECT_FALSE(multiCard[i].state.canDouble);
    EXPECT_NE(twoCard[i].state.hash(), multiCard[i].state.hash());
  }
}

TEST_F(EvaluatorTest, DoubleIsOfferedOnEveryTwoCardTotal) {
  GameRules rules = GameRules::vegasStrip();

  // BlackjackGame::canDoubleDown() allows any two-card hand. An audit that
  // offers DOUBLE only on 9-11 never scores the agent on doubling a 16.
  for (const AuditState &entry : buildAuditStates(rules, HandContext::TwoCard)) {
    EXPECT_NE(std::find(entry.validActions.begin(), entry.validActions.end(),
                        Action::DOUBLE),
              entry.validActions.end())
        << "total=" << entry.state.playerTotal
        << " soft=" << entry.state.hasUsableAce;
  }

  // ...and never on a hand that has already hit.
  for (const AuditState &entry : buildAuditStates(rules, HandContext::MultiCard)) {
    EXPECT_EQ(std::find(entry.validActions.begin(), entry.validActions.end(),
                        Action::DOUBLE),
              entry.validActions.end());
  }
}

TEST_F(EvaluatorTest, BasicStrategyBaselineOutperformsRandomPlay) {
  Evaluator eval(GameRules::vegasStrip());
  BasicStrategyAgent book(/*allowSurrender=*/false);
  RandomAgent chaos(/*seed=*/7);

  auto bookResult = eval.evaluate(&book, 20000, false);
  auto randomResult = eval.evaluate(&chaos, 20000, false);

  // The point of the baseline is to anchor the scale. Book play loses a little;
  // random play loses a lot. If this ever inverts, the reward accounting or the
  // strategy table is wrong.
  EXPECT_GT(bookResult.avgReward, randomResult.avgReward);
  EXPECT_GT(bookResult.avgReward, -0.10);
  EXPECT_LT(randomResult.avgReward, -0.20);
  EXPECT_GT(bookResult.rewardStdErr, 0.0);
}

TEST_F(EvaluatorTest, BustCountSubsetOfLosses) {
  auto result = evaluator.evaluate(agent.get(), 200, false);

  // Busts are a subset of losses
  EXPECT_LE(result.busts, result.losses);
  EXPECT_LE(result.bustRate, result.lossRate + 1e-9);
}

TEST_F(EvaluatorTest, BlackjackCountSubsetOfWins) {
  auto result = evaluator.evaluate(agent.get(), 500, false);

  EXPECT_LE(result.blackjacks, result.wins);
}

// === BasicStrategy correctness ===

TEST_F(EvaluatorTest, BasicStrategyStandOnHard20) {
  const BasicStrategy &bs = evaluator.getBasicStrategy();
  EXPECT_EQ(bs.getAction(State(20, 10, false)), Action::STAND);
}

TEST_F(EvaluatorTest, BasicStrategySurrenderHard16VsDealer10) {
  const BasicStrategy &bs = evaluator.getBasicStrategy();
  // With surrender, optimal play is surrender on hard 16 vs 10
  EXPECT_EQ(bs.getAction(State(16, 10, false)), Action::SURRENDER);
}

TEST_F(EvaluatorTest, BasicStrategyStandOnHard17) {
  const BasicStrategy &bs = evaluator.getBasicStrategy();
  // 17 should stand against any dealer card
  for (int dealer = 2; dealer <= 10; ++dealer) {
    EXPECT_EQ(bs.getAction(State(17, dealer, false)), Action::STAND)
        << "dealer=" << dealer;
  }
}

TEST_F(EvaluatorTest, BasicStrategyStandOnSoft18VsDealer7) {
  const BasicStrategy &bs = evaluator.getBasicStrategy();
  EXPECT_EQ(bs.getAction(State(18, 7, true)), Action::STAND);
}

TEST_F(EvaluatorTest, DoubleSubstitutableByHitOnlyWhenDoublingIsImpossible) {
  const BasicStrategy &bs = evaluator.getBasicStrategy();

  // A hand that has already hit cannot double, so HIT is the correct play
  // wherever the book says DOUBLE.
  State multiCard(11, 6, false, /*canSplit=*/false, /*canDouble=*/false);
  EXPECT_EQ(bs.getAction(multiCard), Action::DOUBLE);
  EXPECT_TRUE(bs.isCorrectAction(multiCard, Action::HIT));

  // On an opening hand it is simply the wrong play. Accepting it here is what
  // let an agent that never learned to double 11 still score as correct.
  State twoCard(11, 6, false, /*canSplit=*/false, /*canDouble=*/true);
  EXPECT_EQ(bs.getAction(twoCard), Action::DOUBLE);
  EXPECT_FALSE(bs.isCorrectAction(twoCard, Action::HIT));
  EXPECT_TRUE(bs.isCorrectAction(twoCard, Action::DOUBLE));
}

// === compareWithBasicStrategy ===

TEST_F(EvaluatorTest, CompareWithBasicStrategyReturnsValidRange) {
  double accuracy = evaluator.compareWithBasicStrategy(agent.get());

  EXPECT_GE(accuracy, 0.0);
  EXPECT_LE(accuracy, 1.0);
}

TEST_F(EvaluatorTest, CompareWithBasicStrategyIsDeterministic) {
  // Exhaustive iteration has no randomness → same result each call
  double accuracy1 = evaluator.compareWithBasicStrategy(agent.get());
  double accuracy2 = evaluator.compareWithBasicStrategy(agent.get());

  EXPECT_DOUBLE_EQ(accuracy1, accuracy2);
}
