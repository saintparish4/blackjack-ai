#include "ai/QLearningAgent.hpp"
#include "training/Trainer.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;

// Global trainer for signal handling
std::unique_ptr<Trainer> g_trainer;

void signalHandler(int signum) {
  std::cout << "\n\nInterrupt signal (" << signum << ") received.\n";

  if (g_trainer) {
    std::cout << "Saving checkpoint before exiting...\n";
    g_trainer->pause();

    // Trainer will auto-save on destruction
    std::cout << "Checkpoint saved. Exiting...\n";
  }

  exit(signum);
}

int main(int argc, char *argv[]) {
  // Register signal handler for graceful shutdown
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);

  std::cout << "========================\n";
  std::cout << "Blackjack Q-Learning Training \n";
  std::cout << "========================\n\n";

  // Parse command line arguments
  size_t numEpisodes = 1'000'000;
  std::string checkpointLoad = "";

  if (argc > 1) {
    numEpisodes = std::stoul(argv[1]);
  }
  if (argc > 2) {
    checkpointLoad = argv[2];
  }

  // Configure Q-Learning agent
  QLearningAgent::Hyperparameters agentParams;
  agentParams.learningRate = 0.1;
  agentParams.discountFactor = 0.95;
  agentParams.epsilon = 1.0;
  agentParams.epsilonDecay = 0.99995;
  agentParams.epsilonMin = 0.01;

  auto agent = std::make_shared<QLearningAgent>(agentParams);

  // Load checkpoint if specified
  if (!checkpointLoad.empty()) {
    std::cout << "Loading checkpoint: " << checkpointLoad << "\n\n";
    agent->load(checkpointLoad);
  }

  // Configure training
  TrainingConfig config;
  config.numEpisodes = numEpisodes;
  config.evalFrequency = 10'000;
  config.evalGames = 1'000;
  config.checkpointFrequency = 50'000;
  config.checkpointDir = "./checkpoints";
  config.logDir = "./logs";
  config.verbose = true;
  config.earlyStoppingPatience = 10;
  config.minImprovement = 0.001;

  // Dealer hits on soft 17 (common casino rule)
  config.gameRules.dealerHitsSoft17 = true;
  config.gameRules.blackjackPayout = 1.5;
  config.gameRules.numDecks = 6;

  // Create trainer
  g_trainer = std::make_unique<Trainer>(agent, config);

  // Set progress callback
  g_trainer->setProgressCallback([](const TrainingMetrics & /*metrics*/) {
    std::cout << std::string(50, '-') << "\n";
  });

  std::cout << "Starting training...\n";
  std::cout << "Press Ctrl+C to stop and save checkpoint\n\n";

  // Start training
  auto startTime = std::chrono::high_resolution_clock::now();

  TrainingMetrics finalMetrics = g_trainer->train();

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);

  // Print final results
  std::cout << "\n";
  std::cout << "==================================\n";
  std::cout << "     Training Complete!           \n";
  std::cout << "==================================\n";
  std::cout << "Total episodes: " << finalMetrics.totalEpisodes << "\n";
  std::cout << "Training time: " << duration.count() << " seconds\n";
  std::cout << "Episodes/sec: "
            << (finalMetrics.totalEpisodes / std::max<decltype(duration)::rep>(1, duration.count()))
            << "\n\n";

  std::cout << "Final Performance:\n";
  std::cout << "  Win rate: " << (finalMetrics.winRate * 100) << "%\n";
  std::cout << "  Loss rate: " << (finalMetrics.lossRate * 100) << "%\n";
  std::cout << "  Push rate: " << (finalMetrics.pushRate * 100) << "%\n";
  std::cout << "  Avg reward: " << finalMetrics.avgReward << "\n";
  std::cout << "  Bust rate: " << (finalMetrics.bustRate * 100) << "%\n\n";

  std::cout << "Learning Progress:\n";
  std::cout << "  States learned: " << finalMetrics.statesLearned << "\n";
  std::cout << "  Final epsilon: " << finalMetrics.currentEpsilon << "\n\n";

  // Save final model
  std::string finalPath = "./models/final_agent";
  agent->save(finalPath);
  std::cout << "Final model saved to: " << finalPath << "\n";

  // Export Q-table for analysis
  agent->exportQTable("./analysis/q_table.csv");
  std::cout << "Q-table exported to: ./analysis/q_table.csv\n";

  std::cout
      << "\nTraining complete. Check logs/ directory for detailed metrics.\n";

  return 0;
}