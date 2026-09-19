#pragma once

#include "../ai/Agent.hpp"
#include "../game/BlackjackGame.hpp"
#include "Evaluator.hpp"
#include "Logger.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace blackjack {
namespace training {

/**
 * @brief Signal early stopping watches for improvement.
 *
 * Win rate used to be the signal. It saturates near 43% within the first few
 * evaluations, because even perfect play loses more hands than it wins, so it
 * said nothing about whether the policy was still improving.
 */
enum class EarlyStopMetric {
  ACCURACY, ///< Policy-audit agreement with basic strategy: no sampling noise
  EV,       ///< Mean reward per round over evalGames: noisy at small evalGames
};

/** Parse "accuracy" / "ev". @throws std::invalid_argument on anything else. */
EarlyStopMetric earlyStopMetricFromString(const std::string &name);
std::string toString(EarlyStopMetric metric);

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

  /// Directory for logs. Empty = no CSV log (Logger creates the directory).
  std::string logDir = "./logs";

  /// Game rules to use
  GameRules gameRules;

  /// Enable verbose logging
  bool verbose = true;

  /// Early stopping: stop if the metric doesn't improve for N evaluations
  size_t earlyStoppingPatience = 10;

  /// Minimum improvement to reset the patience counter, in the metric's own
  /// units: 0.001 is 0.1 percentage points of accuracy, or 0.001 units of EV.
  double minImprovement = 0.001;

  /// What "improvement" means for early stopping.
  EarlyStopMetric earlyStoppingMetric = EarlyStopMetric::ACCURACY;

  /// Seeds the training shoe and the evaluation shoes. The agent's own
  /// exploration RNG is seeded where the agent is built. Unset = nondeterministic.
  std::optional<uint32_t> seed;

  /// Periodic and final checkpoints. Off for experiment sweeps, which save
  /// only the final model.
  bool saveCheckpoints = true;

  /// Rounds per policy for the end-of-run EV comparison against baselines.
  size_t reportEvalRounds = 200'000;

  // ---- Reporting fields (used by saveTrainingReport) ----

  /// Directory for training report output (default: ./analysis)
  std::string reportDir = "./analysis";

  /// Human-readable rule preset name for the report header
  std::string rulesPresetName = "default";

  /// Agent hyperparameters stored here for inclusion in the report
  std::string agentName = "Q-Learning";
  double learningRate   = 0.1;
  double discountFactor = 0.95;
  double epsilon        = 1.0;
  double epsilonDecay   = 0.99995;
  double epsilonMin     = 0.01;
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
  /// Agreement with basic strategy (0-1). Win rate saturates within the first
  /// few evaluations, so this is the metric that actually tracks convergence.
  double strategyAccuracy;

  TrainingMetrics()
      : totalEpisodes(0), avgReward(0.0), winRate(0.0), lossRate(0.0),
        pushRate(0.0), bustRate(0.0), currentEpsilon(0.0), statesLearned(0),
        strategyAccuracy(0.0) {}
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
   * @brief Wall time spent training, excluding the end-of-run report
   */
  double getTrainingSeconds() const { return trainingSeconds_; }

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
   * @brief Request clean stop (saves checkpoint before exiting loop)
   */
  void requestStop() { shouldStop_ = true; }

  /**
   * @brief Check if a stop has been requested
   */
  bool shouldStop() const { return shouldStop_; }

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

  std::atomic<bool> paused_;
  std::atomic<bool> shouldStop_;
  size_t episodesSinceImprovement_;
  double bestMetric_;
  /// Wall time spent inside trainEpisodes(), excluding the end-of-run report's
  /// baseline evaluations, so episodes/sec measures training alone.
  double trainingSeconds_ = 0.0;

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
   * @param outcomes One outcome per player hand (e.g. after split); reward is summed.
   * @param wasDoubledByHand One bool per hand; doubled hands get 2x reward.
   */
  void finishEpisode(std::vector<ai::Experience> &experiences,
                    const std::vector<Outcome> &outcomes,
                    const std::vector<bool> &wasDoubledByHand);

  /**
   * @brief Run exhaustive convergence check against basic strategy, print to
   * stdout (if verbose), and save a full text report to reportDir.
   *
   * Called automatically at the end of train().
   */
  void runAndSaveReport(const TrainingMetrics &finalMetrics);
};
} // namespace training
} // namespace blackjack
