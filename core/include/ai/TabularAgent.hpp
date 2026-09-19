#pragma once

#include "Agent.hpp"
#include "PolicyTable.hpp"
#include <cstdint>
#include <optional>
#include <random>
#include <string>

namespace blackjack {
namespace ai {

/** How the step size α is chosen for each update. */
enum class AlphaSchedule : uint8_t {
  CONSTANT, ///< α = learningRate for every update
  /// α = max(alphaMin, 1 / (1 + n)^alphaOmega), where n is how many updates
  /// this (state, action) has already had. Rare, high-variance actions keep a
  /// large step while they are young and average down their noise as they age.
  VISIT,
};

/** Parse "constant" / "visit". @throws std::invalid_argument otherwise. */
AlphaSchedule alphaScheduleFromString(const std::string &name);
std::string toString(AlphaSchedule schedule);

/**
 * @brief Shared machinery for tabular value-based agents.
 *
 * Q-table with per-action visit counts, ε-greedy exploration decayed once per
 * learn() call, a step-size schedule, and binary save/load. Subclasses differ
 * in how they value the next state (bootstrapValue); Double Q-learning also
 * overrides learn() to keep two tables.
 *
 * Every subclass writes the same .qtable format, so any saved model loads into
 * a QLearningAgent for play: greedy action selection is identical.
 */
class TabularAgent : public Agent {
public:
  struct Hyperparameters {
    double learningRate = 0.1;
    double discountFactor = 0.95;
    double epsilon = 1.0;
    double epsilonDecay = 0.99995;
    double epsilonMin = 0.01;
    AlphaSchedule alphaSchedule = AlphaSchedule::CONSTANT;
    double alphaOmega = 0.8; ///< VISIT only; (0.5, 1] satisfies Robbins-Monro
    double alphaMin = 0.0;   ///< VISIT only; floor on the step size

    bool isValid() const {
      return learningRate > 0 && learningRate <= 1 && discountFactor >= 0 &&
             discountFactor <= 1 && epsilon >= 0 && epsilon <= 1 &&
             epsilonDecay > 0 && epsilonDecay <= 1 && epsilonMin >= 0 &&
             epsilonMin <= epsilon && alphaOmega > 0 && alphaOmega <= 1 &&
             alphaMin >= 0 && alphaMin <= 1;
    }
  };

  /** seed: fixes the exploration stream so a run can be reproduced exactly.
   *  Unseeded agents draw from std::random_device. */
  explicit TabularAgent(const Hyperparameters &params,
                        std::optional<uint32_t> seed = std::nullopt);

  Action chooseAction(const State &state,
                      const std::vector<Action> &validActions,
                      bool training = true) override;
  void learn(const Experience &experience) override;
  double getQValue(const State &state, Action action) const override;
  void save(const std::string &filepath) const override;
  void load(const std::string &filepath) override;
  double getExplorationRate() const override { return epsilon_; }
  size_t getStateCount() const override { return qTable_.size(); }

  uint32_t getVisitCount(const State &state, Action action) const override {
    return qTable_.getVisitCount(state, action);
  }

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
  virtual void reset();

protected:
  /** Value of nextState that the update bootstraps from (not called when the
   *  experience is terminal). validNextActions is never empty. */
  virtual double bootstrapValue(const State &nextState,
                                const std::vector<Action> &validNextActions) const = 0;

  /** Step size for an update to a pair that has had priorUpdates updates. */
  double stepSize(uint32_t priorUpdates) const;

  /** Next-state actions for the update: the experience's, or HIT/STAND when
   *  it carries none. */
  static const std::vector<Action> &
  nextActionsFor(const Experience &experience);

  void decayEpsilon();
  Action greedyAction(const State &state,
                      const std::vector<Action> &validActions) const;

  Hyperparameters params_;
  PolicyTable qTable_; ///< What the agent acts on, audits and saves
  double epsilon_;
  mutable std::mt19937 rng_;
  uint64_t stepCount_;

private:
  Action epsilonGreedy(const State &state,
                       const std::vector<Action> &validActions);
};

} // namespace ai
} // namespace blackjack
