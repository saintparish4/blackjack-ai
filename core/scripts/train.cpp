#include "ai/QLearningAgent.hpp"
#include "game/GameRules.hpp"
#include "training/Trainer.hpp"
#include "util/ConfigParser.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
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

/** Returns a GameRules struct for the named preset (case-insensitive hyphenated
 *  names). Falls back to default GameRules and prints a warning on unknown names. */
static GameRules rulesFromPreset(const std::string& preset) {
  if (preset == "vegas-strip")   return GameRules::vegasStrip();
  if (preset == "downtown")      return GameRules::downtown();
  if (preset == "atlantic-city") return GameRules::atlanticCity();
  if (preset == "european")      return GameRules::european();
  if (preset == "single-deck")   return GameRules::singleDeck();
  std::cerr << "Warning: unknown rules_preset '" << preset
            << "', falling back to default rules.\n";
  return GameRules{};
}

int main(int argc, char *argv[]) {
  signal(SIGINT,  signalHandler);
  signal(SIGTERM, signalHandler);

  std::cout << "========================\n";
  std::cout << "Blackjack Q-Learning Training\n";
  std::cout << "========================\n\n";

  // --- Argument parsing ---
  // Named: --config FILE
  // Positional (legacy): episodes  [checkpoint-path]
  // Priority: positional CLI > config file > built-in defaults
  std::string configFile;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--config" && i + 1 < argc) {
      configFile = argv[++i];
    } else if (arg.rfind("--", 0) != 0) {
      positional.push_back(arg);
    } else {
      std::cerr << "Warning: unrecognised flag '" << arg << "' (ignored).\n";
    }
  }

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

  // --- Episodes and checkpoint (positional args override config) ---
  size_t numEpisodes = static_cast<size_t>(cfg.getInt("episodes", 1'000'000));
  std::string checkpointLoad = "";

  if (!positional.empty()) {
    numEpisodes = std::stoul(positional[0]);
  }
  if (positional.size() >= 2) {
    checkpointLoad = positional[1];
  }

  // --- Q-learning hyperparameters ---
  QLearningAgent::Hyperparameters agentParams;
  agentParams.learningRate  = cfg.getDouble("learning_rate",  0.1);
  agentParams.discountFactor= cfg.getDouble("discount_factor",0.95);
  agentParams.epsilon       = cfg.getDouble("epsilon",        1.0);
  agentParams.epsilonDecay  = cfg.getDouble("epsilon_decay",  0.99995);
  agentParams.epsilonMin    = cfg.getDouble("epsilon_min",    0.01);

  auto agent = std::make_shared<QLearningAgent>(agentParams);

  if (!checkpointLoad.empty()) {
    std::cout << "Loading checkpoint: " << checkpointLoad << "\n\n";
    agent->load(checkpointLoad);
  }

  // --- Game rules (preset then per-field overrides) ---
  std::string preset = cfg.getString("rules_preset", "vegas-strip");
  GameRules gameRules = rulesFromPreset(preset);

  if (cfg.has("num_decks"))
    gameRules.numDecks = static_cast<size_t>(cfg.getInt("num_decks"));
  if (cfg.has("dealer_hits_soft_17"))
    gameRules.dealerHitsSoft17 = cfg.getBool("dealer_hits_soft_17");
  if (cfg.has("surrender"))
    gameRules.surrender = cfg.getBool("surrender");

  // --- Training config ---
  TrainingConfig config;
  config.numEpisodes           = numEpisodes;
  config.evalFrequency         = static_cast<size_t>(cfg.getInt("eval_frequency",      10'000));
  config.evalGames             = static_cast<size_t>(cfg.getInt("eval_games",          1'000));
  config.checkpointFrequency   = static_cast<size_t>(cfg.getInt("checkpoint_frequency",50'000));
  config.checkpointDir         = cfg.getString("checkpoint_dir", "./checkpoints");
  config.logDir                = cfg.getString("log_dir",        "./logs");
  config.verbose               = cfg.getBool("verbose",          true);
  config.earlyStoppingPatience = static_cast<size_t>(cfg.getInt("early_stopping_patience", 10));
  config.minImprovement        = cfg.getDouble("min_improvement", 0.001);
  config.gameRules             = gameRules;

  // --- Create trainer ---
  g_trainer = std::make_unique<Trainer>(agent, config);

  g_trainer->setProgressCallback([](const TrainingMetrics& /*metrics*/) {
    std::cout << std::string(50, '-') << "\n";
  });

  std::cout << "Starting training...\n";
  std::cout << "Press Ctrl+C to stop and save checkpoint\n\n";

  auto startTime = std::chrono::high_resolution_clock::now();
  TrainingMetrics finalMetrics = g_trainer->train();
  auto endTime   = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);

  // --- Final results ---
  std::cout << "\n";
  std::cout << "==================================\n";
  std::cout << "     Training Complete!           \n";
  std::cout << "==================================\n";
  std::cout << "Total episodes:  " << finalMetrics.totalEpisodes << "\n";
  std::cout << "Training time:   " << duration.count() << " seconds\n";
  std::cout << "Episodes/sec:    "
            << (finalMetrics.totalEpisodes /
                std::max<decltype(duration)::rep>(1, duration.count()))
            << "\n\n";

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
  std::string finalPath = "./models/final_agent";
  agent->save(finalPath);
  std::cout << "Final model saved to: " << finalPath << "\n";

  agent->exportQTable("./analysis/q_table.csv");
  std::cout << "Q-table exported to:  ./analysis/q_table.csv\n";

  std::cout << "\nTraining complete. Check logs/ directory for detailed metrics.\n";

  return 0;
}
