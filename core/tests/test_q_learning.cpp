#include "ai/GameStateConverter.hpp"
#include "ai/PolicyTable.hpp"
#include "ai/QLearningAgent.hpp"
#include "ai/State.hpp"
#include "game/BlackjackGame.hpp"
#include <filesystem>
#include <fstream>
#include <set>
#include <gtest/gtest.h>


using namespace blackjack;
using namespace blackjack::ai;

class QLearningTest : public ::testing::Test {
protected:
  QLearningAgent::Hyperparameters params;

  void SetUp() override {
    params.learningRate = 0.1;
    params.discountFactor = 0.9;
    params.epsilon = 1.0;
    params.epsilonDecay = 0.999;
    params.epsilonMin = 0.01;
  }
};

// === State Tests ===

TEST_F(QLearningTest, StateHashIsUnique) {
  State s1(12, 10, false);
  State s2(12, 10, true);  // Different usable ace
  State s3(13, 10, false); // Different player total
  State s4(12, 9, false);  // Different dealer card

  EXPECT_NE(s1.hash(), s2.hash());
  EXPECT_NE(s1.hash(), s3.hash());
  EXPECT_NE(s1.hash(), s4.hash());
}

TEST_F(QLearningTest, StateEqualityWorks) {
  State s1(16, 10, true);
  State s2(16, 10, true);
  State s3(16, 10, false);

  EXPECT_EQ(s1, s2);
  EXPECT_NE(s1, s3);
}

TEST_F(QLearningTest, StateValidation) {
  State valid(16, 10, false);
  EXPECT_TRUE(valid.isValid());

  State invalidPlayerTotal(3, 10, false);
  EXPECT_FALSE(invalidPlayerTotal.isValid());

  State invalidDealerCard(16, 11, false);
  EXPECT_FALSE(invalidDealerCard.isValid());
}

// === PolicyTable Tests ===

TEST_F(QLearningTest, PolicyTableDefaultValue) {
  PolicyTable table(0.5);
  State s(12, 10, false);

  // Unvisited state should return default value
  EXPECT_DOUBLE_EQ(table.get(s, Action::HIT), 0.5);
  EXPECT_DOUBLE_EQ(table.get(s, Action::STAND), 0.5);
}

TEST_F(QLearningTest, PolicyTableSetAndGet) {
  PolicyTable table;
  State s(16, 10, true);

  table.set(s, Action::HIT, 0.75);
  table.set(s, Action::STAND, 0.25);

  EXPECT_DOUBLE_EQ(table.get(s, Action::HIT), 0.75);
  EXPECT_DOUBLE_EQ(table.get(s, Action::STAND), 0.25);
}

TEST_F(QLearningTest, PolicyTableGetMaxAction) {
  PolicyTable table;
  State s(16, 10, false);

  table.set(s, Action::HIT, 0.3);
  table.set(s, Action::STAND, 0.7);
  table.set(s, Action::DOUBLE, 0.1);

  std::vector<Action> validActions = {Action::HIT, Action::STAND,
                                      Action::DOUBLE};

  Action best = table.getMaxAction(s, validActions);
  EXPECT_EQ(best, Action::STAND);
}

TEST_F(QLearningTest, PolicyTableSaveAndLoad) {
  PolicyTable table1;
  State s(18, 9, true);

  table1.set(s, Action::HIT, 0.123);
  table1.set(s, Action::STAND, 0.456);

  std::string filepath =
      (std::filesystem::temp_directory_path() / "test_qtable.bin").string();
  table1.saveToBinary(filepath);

  PolicyTable table2;
  table2.loadFromBinary(filepath);

  EXPECT_DOUBLE_EQ(table2.get(s, Action::HIT), 0.123);
  EXPECT_DOUBLE_EQ(table2.get(s, Action::STAND), 0.456);

  // Cleanup
  std::filesystem::remove(filepath);
}

// === Q-Learning Agent Tests ===

TEST_F(QLearningTest, AgentInitialization) {
  QLearningAgent agent(params);

  EXPECT_EQ(agent.getName(), "Q-Learning");
  EXPECT_DOUBLE_EQ(agent.getEpsilon(), 1.0);
  EXPECT_EQ(agent.getStateSpaceSize(), 0);
}

TEST_F(QLearningTest, AgentChoosesRandomActionWhenExploring) {
  params.epsilon = 1.0; // Always explore
  QLearningAgent agent(params);

  State s(16, 10, false);
  std::vector<Action> validActions = {Action::HIT, Action::STAND};

  // Should choose randomly
  std::map<Action, int> actionCounts;

  for (int i = 0; i < 100; ++i) {
    Action action = agent.chooseAction(s, validActions, true);
    actionCounts[action]++;
  }

  // Both actions should be chosen at least once
  EXPECT_GT(actionCounts[Action::HIT], 0);
  EXPECT_GT(actionCounts[Action::STAND], 0);
}

TEST_F(QLearningTest, AgentExploitsWhenNotTraining) {
  QLearningAgent agent(params);

  State s(16, 10, false);

  // Manually set Q-values to prefer STAND
  Experience exp1(s, Action::HIT, -1.0, State(4, 1, false), true);
  agent.learn(exp1);

  Experience exp2(s, Action::STAND, 1.0, State(4, 1, false), true);
  agent.learn(exp2);

  std::vector<Action> validActions = {Action::HIT, Action::STAND};

  // When not training (exploitation), should always choose STAND
  agent.setEpsilon(0.0); // Force exploitation

  for (int i = 0; i < 10; ++i) {
    Action action = agent.chooseAction(s, validActions, false);
    EXPECT_EQ(action, Action::STAND);
  }
}

TEST_F(QLearningTest, AgentLearnsFromPositiveReward) {
  QLearningAgent agent(params);

  State s(20, 10, false);
  Action a = Action::STAND;

  // Initial Q-value should be 0
  EXPECT_DOUBLE_EQ(agent.getQValue(s, a), 0.0);

  // Give positive reward
  Experience exp(s, a, 1.0, State(4, 1, false), true);
  agent.learn(exp);

  // Q-value should increase
  EXPECT_GT(agent.getQValue(s, a), 0.0);
}

TEST_F(QLearningTest, AgentLearnsFromNegativeReward) {
  QLearningAgent agent(params);

  State s(16, 10, false);
  Action a = Action::HIT;

  // Give negative reward (busted)
  Experience exp(s, a, -1.0, State(4, 1, false), true);
  agent.learn(exp);

  // Q-value should decrease
  EXPECT_LT(agent.getQValue(s, a), 0.0);
}

TEST_F(QLearningTest, EpsilonDecaysOverTime) {
  params.epsilon = 1.0;
  params.epsilonDecay = 0.99;
  params.epsilonMin = 0.1;

  QLearningAgent agent(params);

  EXPECT_DOUBLE_EQ(agent.getEpsilon(), 1.0);

  // Learn from dummy experience multiple times
  State s(12, 10, false);
  Experience exp(s, Action::HIT, 0.0, State(4, 1, false), true);

  for (int i = 0; i < 100; ++i) {
    agent.learn(exp);
  }

  // Epsilon should have decayed
  EXPECT_LT(agent.getEpsilon(), 1.0);
  EXPECT_GE(agent.getEpsilon(), params.epsilonMin);
}

TEST_F(QLearningTest, AgentSaveAndLoad) {
  QLearningAgent agent1(params);

  // Train agent on some experiences
  State s1(16, 10, false);
  State s2(18, 9, true);

  agent1.learn(Experience(s1, Action::HIT, -1.0, State(4, 1, false), true));
  agent1.learn(Experience(s2, Action::STAND, 1.0, State(4, 1, false), true));

  double q1 = agent1.getQValue(s1, Action::HIT);
  double q2 = agent1.getQValue(s2, Action::STAND);

  // Save agent
  std::string filepath =
      (std::filesystem::temp_directory_path() / "test_agent").string();
  agent1.save(filepath);

  // Load into new agent
  QLearningAgent agent2(params);
  agent2.load(filepath);

  // Check Q-values match
  EXPECT_DOUBLE_EQ(agent2.getQValue(s1, Action::HIT), q1);
  EXPECT_DOUBLE_EQ(agent2.getQValue(s2, Action::STAND), q2);

  // Cleanup
  std::filesystem::remove(filepath + ".qtable");
  std::filesystem::remove(filepath + ".meta");
}

// === Convergence Tests ===

TEST_F(QLearningTest, AgentLearnsSimpleStrategy) {
  // Test if agent learns that standing on 20 is better than hitting

  params.epsilon = 0.1; // Low exploration
  params.learningRate = 0.1;
  QLearningAgent agent(params);

  State s(20, 10, false);

  // Simulate training: hitting on 20 always busts (reward -1)
  for (int i = 0; i < 100; ++i) {
    agent.learn(Experience(s, Action::HIT, -1.0, State(4, 1, false), true));
  }

  // Standing on 20 usually wins (reward +1)
  for (int i = 0; i < 100; ++i) {
    agent.learn(Experience(s, Action::STAND, 1.0, State(4, 1, false), true));
  }

  // Agent should prefer STAND
  agent.setEpsilon(0.0); // Pure exploitation
  std::vector<Action> validActions = {Action::HIT, Action::STAND};
  Action chosen = agent.chooseAction(s, validActions, false);

  EXPECT_EQ(chosen, Action::STAND);

  // Verify Q-values
  double qHit = agent.getQValue(s, Action::HIT);
  double qStand = agent.getQValue(s, Action::STAND);

  EXPECT_LT(qHit, qStand) << "Q(STAND) should be higher than Q(HIT)";
}

TEST_F(QLearningTest, AgentLearnsToHitOn16VsTen) {
  // Basic strategy says: hit on hard 16 vs dealer 10

  params.learningRate = 0.1;
  params.epsilon = 0.1;
  QLearningAgent agent(params);

  State s(16, 10, false); // Hard 16 vs 10

  // Simulate: hitting sometimes wins, standing usually loses
  for (int i = 0; i < 50; ++i) {
    // Hit: 30% win, 70% bust
    double reward = (i % 10 < 3) ? 1.0 : -1.0;
    agent.learn(Experience(s, Action::HIT, reward, State(4, 1, false), true));
  }

  for (int i = 0; i < 50; ++i) {
    // Stand: almost always lose to dealer
    agent.learn(Experience(s, Action::STAND, -1.0, State(4, 1, false), true));
  }

  // Agent should prefer HIT (higher expected value)
  agent.setEpsilon(0.0);
  std::vector<Action> validActions = {Action::HIT, Action::STAND};
  Action chosen = agent.chooseAction(s, validActions, false);

  EXPECT_EQ(chosen, Action::HIT);
}

// === Integration Tests with BlackjackGame ===

TEST_F(QLearningTest, IntegrationWithGameEngine) {
  BlackjackGame game;
  QLearningAgent agent(params);

  // Play one round
  game.startRound();

  const Hand &playerHand = game.getPlayerHand();
  const Hand &dealerHand = game.getDealerHand(true); // Hide hole card

  // Convert to AI state
  State state = GameStateConverter::toAIState(playerHand, dealerHand);

  EXPECT_TRUE(state.isValid());
  EXPECT_GE(state.playerTotal, 4);
  EXPECT_LE(state.playerTotal, 21);
  EXPECT_GE(state.dealerUpCard, 1);
  EXPECT_LE(state.dealerUpCard, 10);
}

TEST_F(QLearningTest, RewardConversionIsCorrect) {
  EXPECT_DOUBLE_EQ(
      GameStateConverter::outcomeToReward(Outcome::PLAYER_BLACKJACK), 1.5);
  EXPECT_DOUBLE_EQ(GameStateConverter::outcomeToReward(Outcome::PLAYER_WIN),
                   1.0);
  EXPECT_DOUBLE_EQ(GameStateConverter::outcomeToReward(Outcome::PUSH), 0.0);
  EXPECT_DOUBLE_EQ(GameStateConverter::outcomeToReward(Outcome::PLAYER_BUST),
                   -1.0);
  EXPECT_DOUBLE_EQ(GameStateConverter::outcomeToReward(Outcome::SURRENDER),
                   -0.5);
}

TEST_F(QLearningTest, DoubleDownRewardMultiplier) {
  EXPECT_DOUBLE_EQ(
      GameStateConverter::outcomeToReward(Outcome::PLAYER_WIN, true), 2.0);
  EXPECT_DOUBLE_EQ(
      GameStateConverter::outcomeToReward(Outcome::DEALER_WIN, true), -2.0);
  EXPECT_DOUBLE_EQ(
      GameStateConverter::outcomeToReward(Outcome::PUSH, true), 0.0);
}
// === Visit tracking ===
//
// These lock down the distinction that the policy audit depends on: an action
// the agent evaluated and disliked versus one it was never offered. Both leave
// a Q-value; only the first is a decision.

TEST_F(QLearningTest, VisitCountsSeparateUntriedActionsFromDislikedOnes) {
  PolicyTable table;
  State s(11, 6, false, /*canSplit=*/false, /*canDouble=*/true);

  EXPECT_EQ(table.getVisitCount(s, Action::HIT), 0u);
  EXPECT_FALSE(table.wasActionTried(s, Action::DOUBLE));

  table.set(s, Action::HIT, -0.4);

  // HIT was tried and scored badly; DOUBLE was never tried. Both read 0.0 and
  // -0.4 respectively, but only the visit count tells them apart -- and an
  // untried DOUBLE at the default 0.0 would win a greedy argmax here.
  EXPECT_EQ(table.getVisitCount(s, Action::HIT), 1u);
  EXPECT_TRUE(table.wasActionTried(s, Action::HIT));
  EXPECT_EQ(table.getVisitCount(s, Action::DOUBLE), 0u);
  EXPECT_FALSE(table.wasActionTried(s, Action::DOUBLE));
  EXPECT_DOUBLE_EQ(table.get(s, Action::DOUBLE), 0.0);
}

TEST_F(QLearningTest, VisitCountsAccumulateAndSurviveSaveLoad) {
  const std::string path = "test_visits.qtable";

  PolicyTable table;
  State s(16, 10, false, /*canSplit=*/false, /*canDouble=*/true);
  for (int i = 0; i < 5; ++i) {
    table.set(s, Action::HIT, -0.5);
  }
  table.set(s, Action::STAND, -0.6);
  ASSERT_EQ(table.getVisitCount(s, Action::HIT), 5u);

  table.saveToBinary(path);

  PolicyTable restored;
  restored.loadFromBinary(path);

  EXPECT_EQ(restored.getVisitCount(s, Action::HIT), 5u);
  EXPECT_EQ(restored.getVisitCount(s, Action::STAND), 1u);
  EXPECT_EQ(restored.getVisitCount(s, Action::DOUBLE), 0u);
  EXPECT_DOUBLE_EQ(restored.get(s, Action::HIT), -0.5);

  std::filesystem::remove(path);
}

TEST_F(QLearningTest, TwoCardAndMultiCardStatesAreDistinctTableSlots) {
  PolicyTable table;
  State twoCard(11, 6, false, /*canSplit=*/false, /*canDouble=*/true);
  State multiCard(11, 6, false, /*canSplit=*/false, /*canDouble=*/false);

  ASSERT_NE(twoCard.hash(), multiCard.hash());

  table.set(twoCard, Action::DOUBLE, 0.9);

  // Doubling is only ever legal on a two-card hand, so the multi-card slot can
  // never learn DOUBLE. Reading the policy from that slot -- as the audit,
  // chart and CSV export all used to -- shows an agent that never doubles.
  EXPECT_DOUBLE_EQ(table.get(twoCard, Action::DOUBLE), 0.9);
  EXPECT_DOUBLE_EQ(table.get(multiCard, Action::DOUBLE), 0.0);
  EXPECT_FALSE(table.wasActionTried(multiCard, Action::DOUBLE));
}

TEST_F(QLearningTest, CsvExportKeysRowsByFullStateWithoutDuplicates) {
  const std::string path = "test_export.csv";

  PolicyTable table;
  State twoCard(11, 6, false, /*canSplit=*/false, /*canDouble=*/true);
  State multiCard(11, 6, false, /*canSplit=*/false, /*canDouble=*/false);
  table.set(twoCard, Action::DOUBLE, 0.9);
  table.set(multiCard, Action::HIT, 0.2);

  table.exportToCSV(path);

  std::ifstream file(path);
  ASSERT_TRUE(file.is_open());

  std::string header;
  std::getline(file, header);
  EXPECT_NE(header.find("can_double"), std::string::npos);
  EXPECT_NE(header.find("N_DOUBLE"), std::string::npos);

  // Both rows describe player 11 vs 6, so omitting canSplit/canDouble from the
  // key emitted two rows that looked identical and held different policies.
  std::set<std::string> keys;
  std::string line;
  size_t rows = 0;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    ++rows;
    size_t fifthComma = 0;
    for (int i = 0; i < 5; ++i) {
      fifthComma = line.find(',', fifthComma + (i == 0 ? 0 : 1));
    }
    keys.insert(line.substr(0, fifthComma));
  }

  EXPECT_EQ(rows, 2u);
  EXPECT_EQ(keys.size(), rows) << "state key does not uniquely identify a row";

  file.close();
  std::filesystem::remove(path);
}
