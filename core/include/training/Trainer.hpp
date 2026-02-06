#pragma once

#include "../ai/Agent.hpp"
#include "../game/BlackjackGame.hpp"
#include "Evaluator.hpp"
#include "Logger.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace blackjack {
namespace training {
/**
 * @brief Configuration for training session
 */
struct TrainingConfig {
  /// Total number of episodes to train
  size_t numEpisodes = 1'000'000;

  /// Evaluate agent every N episodes
  size_t evalFrequency = 10'000;

  /// Number of games for each evaluation
  size_t evalGames = 1'000;

  /// Save checkpoint every N episodes
  size_t checkpointFrequency = 50'000;

  /// Directory for checkpoints
  std::string checkpointDir = "./checkpoints";

  /// Directory for logs
  std::string logDir = "./logs";

  /// Game rules to use
  GameRules gameRules;

  /// Enable verbose logging
  bool verbose = true;

  /// Early stopping: stop if win rate doesn't improve for N evaluations
  size_t earlyStoppingPatience = 10;

  /// Minimum improvement to reset patience counter (0.1 = 0.1%)
  double minImprovement = 0.001;
};

/**
 * @brief Training statistics for a single episode
 */
struct EpisodeStats {
  size_t episodeNumber;
  int handsPlayed;
  double reward;
  Outcome outcome;
  bool playerBusted;
  bool dealerBusted;

  EpisodeStats()
      : episodeNumber(0), handsPlayed(0), reward(0.0), outcome(Outcome::PUSH),
        playerBusted(false), dealerBusted(false) {}
};

/**
 * @brief Aggregated training metrics
 */
struct TrainingMetrics {
  size_t totalEpisodes;
  double avgReward;
  double winRate;
  double lossRate;
  double pushRate;
  double bustRate;
  double currentEpsilon;
  size_t statesLearned;

  TrainingMetrics()
      : totalEpisodes(0), avgReward(0.0), winRate(0.0), lossRate(0.0),
        pushRate(0.0), bustRate(0.0), currentEpsilon(0.0), statesLearned(0) {}
};

/**
 * @brief Main training engine for RL agents
 *
 * Handles:
 * - Episode execution
 * - Experience collection
 * - Periodic evaluation
 * - Checkpointing
 * - Training metrics
 */
class Trainer {
public:
  /**
   * @brief Construct trainer with agent and config
   */
  Trainer(std::shared_ptr<ai::Agent> agent, const TrainingConfig &config);

  /**
   * @brief Run complete training session
   *
   * @return Final training metrics
   */
  TrainingMetrics train();

  /**
   * @brief Train for specified number of episodes
   *
   * Useful for incremental training
   */
  TrainingMetrics trainEpisodes(size_t numEpisodes);

  /**
   * @brief Run a single training episode
   *
   * @return Episode statistics
   */
  EpisodeStats runEpisode();

  /**
   * @brief Get current training metrics
   */
  TrainingMetrics getMetrics() const { return currentMetrics_; }

  /**
   * @brief Get training history
   */
  const std::vector<TrainingMetrics> &getHistory() const {
    return trainingHistory_;
  }

  /**
   * @brief Set callback for progress updates
   *
   * Called after each evaluation
   */
  void
  setProgressCallback(std::function<void(const TrainingMetrics &)> callback) {
    progressCallback_ = callback;
  }

  /**
   * @brief Pause training (can be resumed)
   */
  void pause() { paused_ = true; }

  /**
   * @brief Resume paused training
   */
  void resume() { paused_ = false; }

  /**
   * @brief Check if training should stop early
   */
  bool shouldStopEarly() const;

private:
  std::shared_ptr<ai::Agent> agent_;
  TrainingConfig config_;
  std::unique_ptr<BlackjackGame> game_;
  std::unique_ptr<Evaluator> evaluator_;
  std::unique_ptr<Logger> logger_;

  TrainingMetrics currentMetrics_;
  std::vector<TrainingMetrics> trainingHistory_;

  std::function<void(const TrainingMetrics &)> progressCallback_;

  bool paused_;
  size_t episodesSinceImprovement_;
  double bestWinRate_;

  /**
   * @brief Execute evaluation
   */
  void evaluate();

  /**
   * @brief Save checkpoint
   */
  void saveCheckpoint(size_t episodeNum);

  /**
   * @brief Update running metrics
   */
  void updateMetrics(const EpisodeStats &stats);

  /**
   * @brief Play agent's turn in episode
   */
  void playAgentTurn(std::vector<ai::Experience> &experiences);

  /**
   * @brief Complete episode and learn from experiences
   */
  void finishEpisode(std::vector<ai::Experience> &experiences, Outcome outcome);
};
} // namespace training
} // namespace blackjack
