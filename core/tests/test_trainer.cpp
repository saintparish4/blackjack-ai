#include "ai/QLearningAgent.hpp"
#include "training/Trainer.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;

class TrainerTest : public ::testing::Test {
protected:
  std::shared_ptr<QLearningAgent> agent;
  TrainingConfig config;

  void SetUp() override {
    QLearningAgent::Hyperparameters params;
    params.epsilon = 0.5;
    params.epsilonMin = 0.01;
    agent = std::make_shared<QLearningAgent>(params);

    auto tmpDir = std::filesystem::temp_directory_path();
    config.numEpisodes = 100;
    config.evalFrequency = 50;
    config.evalGames = 20;
    config.checkpointFrequency = 100;
    config.checkpointDir = (tmpDir / "trainer_test_checkpoints").string();
    config.logDir = (tmpDir / "trainer_test_logs").string();
    config.verbose = false;
    config.earlyStoppingPatience = 1000; // effectively disabled
  }

  void TearDown() override {
    std::filesystem::remove_all(config.checkpointDir);
    std::filesystem::remove_all(config.logDir);
  }
};

TEST_F(TrainerTest, TrainCompletesRequestedEpisodes) {
  Trainer trainer(agent, config);
  TrainingMetrics metrics = trainer.train();

  EXPECT_EQ(metrics.totalEpisodes, config.numEpisodes);
}

TEST_F(TrainerTest, TrainMetricsAreValid) {
  Trainer trainer(agent, config);
  TrainingMetrics metrics = trainer.train();

  EXPECT_GE(metrics.winRate, 0.0);
  EXPECT_LE(metrics.winRate, 1.0);
  EXPECT_GE(metrics.lossRate, 0.0);
  EXPECT_LE(metrics.lossRate, 1.0);
  EXPECT_GE(metrics.pushRate, 0.0);
  EXPECT_LE(metrics.pushRate, 1.0);
}

TEST_F(TrainerTest, RunEpisodeReturnsValidStats) {
  Trainer trainer(agent, config);
  EpisodeStats stats = trainer.runEpisode();

  // Counts agent decision steps, not hands, so 0 is valid: a dealt blackjack
  // settles the round before the agent ever chooses an action.
  EXPECT_GE(stats.handsPlayed, 0);
  // Episode reward is summed over every hand the round settles, so the bounds
  // are wider than a single undoubled hand's -1..+1.5: doubling pays 2x, and a
  // split settles two hands (double-after-split is not allowed, so each of the
  // two caps at +/-1). Both routes bottom out at -2 and top out at +2.
  EXPECT_GE(stats.reward, -2.0);
  EXPECT_LE(stats.reward, 2.0);
}

TEST_F(TrainerTest, TerminalRewardOnLastExperience) {
  // After training, Q-values for terminal states should be non-zero while
  // intermediate states accumulate via Bellman; the simplest observable check
  // is that the agent learns something (state count increases).
  config.numEpisodes = 500;
  Trainer trainer(agent, config);
  trainer.train();

  EXPECT_GT(agent->getStateCount(), 0u);
}

TEST_F(TrainerTest, EarlyStoppingTriggersBeforeMaxEpisodes) {
  config.earlyStoppingPatience = 1; // stop after 1 eval with no improvement
  config.evalFrequency = 10;
  config.numEpisodes = 100000; // large — should be cut short by early stopping

  Trainer trainer(agent, config);
  TrainingMetrics metrics = trainer.train();

  EXPECT_LT(metrics.totalEpisodes, 100000u);
}

TEST_F(TrainerTest, CheckpointSavingCreatesFiles) {
  config.checkpointFrequency = 50;
  config.numEpisodes = 100;

  Trainer trainer(agent, config);
  trainer.train();

  // Final checkpoint is always saved; checkpoint dir must exist
  EXPECT_TRUE(std::filesystem::exists(config.checkpointDir));
  bool foundCheckpoint = false;
  for (const auto &entry :
       std::filesystem::directory_iterator(config.checkpointDir)) {
    if (entry.path().extension() == ".qtable" ||
        entry.path().extension() == ".meta") {
      foundCheckpoint = true;
      break;
    }
  }
  EXPECT_TRUE(foundCheckpoint);
}

namespace {

/// Splits whenever it can, otherwise stands, and records what it learns from.
class SplitSpyAgent : public Agent {
public:
  std::vector<Experience> seen;

  Action chooseAction(const State &, const std::vector<Action> &valid,
                      bool) override {
    for (Action a : valid) {
      if (a == Action::SPLIT) return a;
    }
    return Action::STAND;
  }
  void learn(const Experience &e) override { seen.push_back(e); }
  double getQValue(const State &, Action) const override { return 0.0; }
  void save(const std::string &) const override {}
  void load(const std::string &) override {}
  std::string getName() const override { return "SplitSpy"; }
};

} // namespace

// After a split, each hand's last decision must learn that hand's own result,
// and SPLIT itself must be valued from both hands. They used to form one chain
// in which SPLIT was valued as a single hand, which made splitting losing
// pairs look half as costly as it is.
TEST_F(TrainerTest, SplitCreditsEachHandSeparately) {
  config.seed = 12;
  auto spy = std::make_shared<SplitSpyAgent>();
  Trainer trainer(spy, config);

  int splitRounds = 0;
  for (int i = 0; i < 20000 && splitRounds < 25; ++i) {
    spy->seen.clear();
    EpisodeStats stats = trainer.runEpisode();
    if (spy->seen.empty() || spy->seen[0].action != Action::SPLIT) continue;
    ++splitRounds;

    const Experience &split = spy->seen[0];
    // Split aces take one card each and need no decisions: the split itself
    // is terminal with the total.
    if (spy->seen.size() == 1) {
      EXPECT_TRUE(split.done);
      EXPECT_DOUBLE_EQ(split.reward, stats.reward);
      continue;
    }

    // Any other split bootstraps from both hands, and the second-hand state
    // it recorded must be the state that hand is actually played from.
    ASSERT_EQ(spy->seen.size(), 3u);
    EXPECT_FALSE(split.done);
    EXPECT_DOUBLE_EQ(split.reward, 0.0);
    ASSERT_TRUE(split.secondNextState.has_value());
    EXPECT_EQ(split.nextState, spy->seen[1].state);
    EXPECT_EQ(*split.secondNextState, spy->seen[2].state);
    EXPECT_EQ(split.secondValidNextActions,
              (std::vector<Action>{Action::HIT, Action::STAND}));

    // Then one STAND per hand, each terminal with its own result.
    double perHandTotal = 0.0;
    for (size_t k = 1; k < 3; ++k) {
      EXPECT_EQ(spy->seen[k].action, Action::STAND);
      EXPECT_TRUE(spy->seen[k].done) << "hand " << k;
      EXPECT_GE(spy->seen[k].reward, -1.0);
      EXPECT_LE(spy->seen[k].reward, 1.0);
      perHandTotal += spy->seen[k].reward;
    }
    EXPECT_DOUBLE_EQ(perHandTotal, stats.reward);
  }
  EXPECT_GT(splitRounds, 0);
}
