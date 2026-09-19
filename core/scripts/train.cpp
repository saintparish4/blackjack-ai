#include "ai/QLearningAgent.hpp"
#include "game/GameRules.hpp"
#include "training/RunConfig.hpp"
#include "training/Trainer.hpp"
#include "util/ArgParser.hpp"
#include "util/ConfigParser.hpp"
#include <algorithm>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;
using namespace blackjack::util;

// Global trainer for signal handling
std::unique_ptr<Trainer> g_trainer;

void signalHandler(int signum) {
  std::cout << "\n\nInterrupt signal (" << signum << ") received.\n";
  if (g_trainer) {
    std::cout << "Requesting clean stop...\n";
    g_trainer->requestStop();
  }
}

int main(int argc, char *argv[]) {
  signal(SIGINT,  signalHandler);
  signal(SIGTERM, signalHandler);

  std::cout << "========================\n";
  std::cout << "Blackjack Q-Learning Training\n";
  std::cout << "========================\n\n";

  // --- Argument parsing (named flags only; CLI overrides config) ---
  ArgParser args("train", "Blackjack Q-Learning Trainer");
  args.addFlag("episodes", "e", "Number of training episodes", "1000000");
  args.addFlag("checkpoint", "c", "Resume from checkpoint file", "");
  args.addFlag("config", "", "Load INI config file", "");
  args.addFlag("rules", "r", "Rule preset name", "vegas-strip");
  args.addFlag("seed", "s", "Seed for a reproducible run (agent, shoe and evaluation)", "");
  args.addBool("verbose", "v", "Enable verbose output");
  args.addBool("help", "h", "Show this help message");
  if (!args.parse(argc, argv)) return 0;

  std::string configFile;
  if (args.has("config")) configFile = args.getString("config");

  // --- Load config file ---
  ConfigParser cfg;
  if (!configFile.empty()) {
    try {
      cfg.load(configFile);
      std::cout << "Config loaded: " << configFile << "\n\n";
    } catch (const std::exception& e) {
      std::cerr << "Error loading config: " << e.what() << "\n";
      return 1;
    }
  }

  // CLI > config > default. wasProvided(), not has(): --episodes and --rules
  // are registered with defaults, so has() is true even when they are absent
  // from argv, which would let a built-in default override the config file.
  RunOverrides overrides;
  RunConfig run;
  try {
    if (args.wasProvided("episodes"))
      overrides.episodes = std::stoul(args.getString("episodes"));
    if (args.wasProvided("rules")) overrides.rulesPreset = args.getString("rules");
    if (args.wasProvided("seed"))
      overrides.seed = static_cast<uint32_t>(std::stoul(args.getString("seed")));
    run = loadRunConfig(cfg, overrides);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  TrainingConfig &config = run.training;
  if (args.has("verbose")) config.verbose = true;

  auto agent = makeAgent(run, config.seed);

  std::string checkpointLoad;
  if (args.has("checkpoint")) checkpointLoad = args.getString("checkpoint");
  if (!checkpointLoad.empty()) {
    std::cout << "Loading checkpoint: " << checkpointLoad << "\n\n";
    agent->load(checkpointLoad);
  }

  // --- Create trainer ---
  g_trainer = std::make_unique<Trainer>(agent, config);

  g_trainer->setProgressCallback([](const TrainingMetrics& /*metrics*/) {
    std::cout << std::string(50, '-') << "\n";
  });

  std::cout << "Starting training...\n";
  std::cout << "Press Ctrl+C to stop and save checkpoint\n\n";

  TrainingMetrics finalMetrics = g_trainer->train();

  // Fractional seconds: whole seconds printed a 1.7s run as "1 seconds".
  const double seconds = g_trainer->getTrainingSeconds();

  // --- Final results ---
  std::cout << "\n";
  std::cout << "==================================\n";
  std::cout << "     Training Complete!           \n";
  std::cout << "==================================\n";
  std::cout << "Total episodes:  " << finalMetrics.totalEpisodes << "\n";
  std::cout << "Training time:   " << std::fixed << std::setprecision(2)
            << seconds << " s\n";
  std::cout << "Episodes/sec:    " << std::setprecision(0)
            << (static_cast<double>(finalMetrics.totalEpisodes) /
                std::max(seconds, 1e-9))
            << "\n\n";
  std::cout << std::setprecision(4);

  std::cout << "Final Performance:\n";
  std::cout << "  Win rate:   " << (finalMetrics.winRate  * 100) << "%\n";
  std::cout << "  Loss rate:  " << (finalMetrics.lossRate * 100) << "%\n";
  std::cout << "  Push rate:  " << (finalMetrics.pushRate * 100) << "%\n";
  std::cout << "  Avg reward: " << finalMetrics.avgReward  << "\n";
  std::cout << "  Bust rate:  " << (finalMetrics.bustRate * 100) << "%\n\n";

  std::cout << "Learning Progress:\n";
  std::cout << "  States learned: " << finalMetrics.statesLearned << "\n";
  std::cout << "  Final epsilon:  " << finalMetrics.currentEpsilon << "\n\n";

  // --- Save final model ---
  // Training has already completed at this point, so a write failure must be
  // reported and survived rather than thrown out of main(): letting it
  // terminate() discards the entire run.
  int exitCode = 0;

  const std::string finalPath = "./models/final_agent";
  try {
    agent->save(finalPath);
    std::cout << "Final model saved to: " << finalPath << "\n";
  } catch (const std::exception &e) {
    std::cerr << "ERROR: could not save model to " << finalPath << ": "
              << e.what() << "\n";
    exitCode = 1;
  }

  const std::string qTablePath = "./analysis/q_table.csv";
  try {
    agent->exportQTable(qTablePath);
    std::cout << "Q-table exported to:  " << qTablePath << "\n";
  } catch (const std::exception &e) {
    std::cerr << "ERROR: could not export Q-table to " << qTablePath << ": "
              << e.what() << "\n";
    exitCode = 1;
  }

  std::cout << "\nTraining complete. Check logs/ directory for detailed metrics.\n";

  return exitCode;
}
