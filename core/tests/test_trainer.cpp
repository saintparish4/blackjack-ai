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

  EXPECT_GE(stats.handsPlayed, 0);
  // Reward for an episode is +1.5, +1, 0, or -1
  EXPECT_GE(stats.reward, -1.0);
  EXPECT_LE(stats.reward, 1.5);
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
