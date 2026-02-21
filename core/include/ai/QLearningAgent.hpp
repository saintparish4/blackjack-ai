#pragma once

#include "Agent.hpp"
#include "PolicyTable.hpp"
#include <cstdint>
#include <random>

namespace blackjack {
namespace ai {

/** Q-learning agent: Q(s,a) ← Q + α[R + γ max Q(s',a') - Q]; ε-greedy
 * exploration with decay. */
class QLearningAgent : public Agent {
public:
  struct Hyperparameters {
    double learningRate = 0.1;
    double discountFactor = 0.95;
    double epsilon = 1.0;
    double epsilonDecay = 0.99995;
    double epsilonMin = 0.01;

    bool isValid() const {
      return learningRate > 0 && learningRate <= 1 && discountFactor >= 0 &&
             discountFactor <= 1 && epsilon >= 0 && epsilon <= 1 &&
             epsilonDecay > 0 && epsilonDecay <= 1 && epsilonMin >= 0 &&
             epsilonMin <= epsilon;
    }
  };

  explicit QLearningAgent(const Hyperparameters &params = Hyperparameters{
                              0.1, 0.95, 1.0, 0.99995, 0.01});

  Action chooseAction(const State &state,
                      const std::vector<Action> &validActions,
                      bool training = true) override;
  void learn(const Experience &experience) override;
  double getQValue(const State &state, Action action) const override;
  void save(const std::string &filepath) const override;
  void load(const std::string &filepath) override;
  std::string getName() const override { return "Q-Learning"; }
  double getExplorationRate() const override { return epsilon_; }
  size_t getStateCount() const override { return qTable_.size(); }

  PolicyTable::QValues getAllQValues(const State &state) const {
    return qTable_.getAll(state);
  }
  double getEpsilon() const { return epsilon_; }
  void setEpsilon(double epsilon) {
    epsilon_ = std::max(params_.epsilonMin, std::min(1.0, epsilon));
  }
  size_t getStateSpaceSize() const { return qTable_.size(); }
  const Hyperparameters &getHyperparameters() const { return params_; }
  void exportQTable(const std::string &filepath) const {
    qTable_.exportToCSV(filepath);
  }
  void reset();

private:
  Hyperparameters params_;
  PolicyTable qTable_;
  double epsilon_;
  mutable std::mt19937 rng_;
  uint64_t stepCount_;

  Action epsilonGreedy(const State &state,
                       const std::vector<Action> &validActions);
  Action greedyAction(const State &state,
                      const std::vector<Action> &validActions) const;
  void decayEpsilon();
};
} // namespace ai
} // namespace blackjack