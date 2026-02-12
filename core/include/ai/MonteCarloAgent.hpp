#pragma once

#include "ai/Agent.hpp"
#include "ai/PolicyTable.hpp"
#include <random>
#include <unordered_map>
#include <vector>

namespace blackjack {
namespace ai {
/**
 * @brief Monte Carlo agent for blackjack
 *
 * Implements First-Visit Monte Carlo Control with ε-greedy policy.
 *
 * Key differences from Q-Learning:
 * - Q-Learning: Updates after each step (TD learning)
 * - Monte Carlo: Updates after complete episode (no bootstrapping)
 *
 * Algorithm:
 * 1. Generate episode using current policy
 * 2. For each (state, action) pair in episode:
 *    - Calculate return (sum of future rewards)
 *    - Update Q-value as average of all returns
 * 3. Improve policy (ε-greedy)
 *
 * Advantages:
 * - No bootstrapping bias
 * - Can learn from actual experience
 *
 * Disadvantages:
 * - Requires complete episodes
 * - Higher variance than TD methods
 * - Slower convergence
 */
class MonteCarloAgent : public Agent {
public:
  /**
   * @brief Hyperparameters for Monte Carlo
   */
  struct Hyperparameters {
    double epsilon = 1.0;          //< Exploration rate
    double epsilonDecay = 0.99995; //< Decay per episode
    double epsilonMin = 0.01;      //< Minimum exploration

    bool useFirstVist = true; //< Use first-visit vs every-visit

    bool isValid() const {
      return epsilon >= 0 && epsilon <= 1 && epsilonDecay > 0 &&
             epsilonDecay <= 1 && epsilonMin >= 0 && epsilonMin <= epsilon;
    }
  };

  /**
   * @brief Construct Monte Carlo Agent
   */
  explicit MonteCarloAgent(const Hyperparameters &params = Hyperparameters{
                               1.0, 0.99995, 0.01, true});

  // Agent interface
  Action chooseAction(const State &state,
                      const std::vector<Action> &validActions,
                      bool training = true) override;

  void learn(const Experience &experience) override;

  double getQValue(const State &state, Action action) const override;

  void save(const std::string &filepath) const override;
  void load(const std::string &filepath) override;

  std::string getName() const override { return "Monte Carlo"; }

  // Monte Carlo specific methods
  /**
   * @brief Start a new episode
   *
   * Must be called before collecting experiences for an episode
   */
  void startEpisode();

  /**
   * @brief Finish episode and learn from collected experiences
   *
   * @param finalReward The final reward of the episode
   */
  void finishEpisode(double finalReward);

  /**
   * @brief Get all Q-values for a state
   */
  std::array<double, 4> getAllQValues(const State &state) const {
    return qTable_.getAll(state);
  }

  /**
   * @brief Get current epsilon
   */
  double getEpsilon() const { return epsilon_; }

  /**
   * @brief Set epsilon manually
   */
  void setEpsilon(double epsilon) {
    epsilon_ = std::max(params_.epsilonMin, std::min(1.0, epsilon));
  }

  /**
   * @brief Get number of states learned
   */
  size_t getStateSpaceSize() const { return qTable_.size(); }

  /**
   * @brief Get hyperparameters
   */
  const Hyperparameters &getHyperparameters() const { return params_; }

  /**
   * @brief Export Q-table to CSV
   */
  void exportQTable(const std::string &filepath) const {
    qTable_.exportToCSV(filepath);
  }

  /**
   * @brief Reset learning
   */
  void reset();

  /**
   * @brief Get number of episodes trained
   */
  uint64_t getEpisodeCount() const { return episodeCount_; }

private:
  /**
   * @brief State-action pair for episode tracking
   */
  struct StateActionPair {
    State state;
    Action action;
    size_t stepNumber; // Position in episode

    bool operator==(const StateActionPair &other) const {
      return state == other.state && action == other.action;
    }
  };

  struct StateActionHash {
    size_t operator()(const StateActionPair &sa) const {
      return sa.state.hash() ^ (static_cast<size_t>(sa.action) << 16);
    }
  };

  /**
   * @brief Visit count for averaging returns
   */
  struct VisitInfo {
    double sumReturns = 0.0;
    size_t visitCount = 0;
  };

  Hyperparameters params_;
  PolicyTable qTable_;
  double epsilon_;
  mutable std::mt19937 rng_;

  // Episode tracking
  std::vector<StateActionPair> currentEpisode_;
  std::unordered_map<StateActionPair, VisitInfo, StateActionHash> returns_;

  uint64_t episodeCount_;

  /**
   * @brief ε-greedy action selection
   */
  Action epsilonGreedy(const State &state,
                       const std::vector<Action> &validActions);

  /**
   * @brief Greedy action selection
   */
  Action greedyAction(const State &state,
                      const std::vector<Action> &validActions) const;

  /**
   * @brief Decay epsilon
   */
  void decayEpsilon();

  /**
   * @brief Update Q-values from episode
   */
  void updateFromEpisode(double finalReward);
};
} // namespace ai
} // namespace blackjack