#include "ai/QLearningAgent.hpp"
#include "training/Evaluator.hpp"
#include <gtest/gtest.h>

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

// === Metrics correctness ===

TEST_F(EvaluatorTest, WinLossPushCountsSumToGamesPlayed) {
  auto result = evaluator.evaluate(agent.get(), 100, false);

  EXPECT_EQ(result.gamesPlayed, 100u);
  EXPECT_EQ(result.wins + result.losses + result.pushes, 100u);
}

TEST_F(EvaluatorTest, RatesMatchCounts) {
  auto result = evaluator.evaluate(agent.get(), 200, false);

  EXPECT_NEAR(result.winRate,
              static_cast<double>(result.wins) / result.gamesPlayed, 1e-9);
  EXPECT_NEAR(result.lossRate,
              static_cast<double>(result.losses) / result.gamesPlayed, 1e-9);
  EXPECT_NEAR(result.pushRate,
              static_cast<double>(result.pushes) / result.gamesPlayed, 1e-9);
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

TEST_F(EvaluatorTest, BasicStrategyDoubleSubstitutableByHit) {
  const BasicStrategy &bs = evaluator.getBasicStrategy();
  // Basic strategy may say DOUBLE on 11; HIT should be accepted as correct
  State s(11, 6, false);
  EXPECT_TRUE(bs.isCorrectAction(s, Action::HIT));
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
