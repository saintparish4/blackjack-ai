#pragma once

#include "TabularAgent.hpp"

namespace blackjack {
namespace ai {

/**
 * Double Q-learning (van Hasselt, 2010).
 *
 * Keeps two independent estimates, A and B. Each update picks one at random,
 * chooses the next action by its own argmax, and values that action with the
 * other table. A single table uses the same noisy numbers to pick the max and
 * to value it, which biases it upward, and the bias is largest on the noisiest
 * actions, which here is DOUBLE.
 *
 * The inherited table holds the average of A and B, with visit counts equal
 * to the total updates. Acting, auditing and saving all use it, so a saved
 * model is an ordinary .qtable that any tabular agent can load.
 */
class DoubleQLearningAgent : public TabularAgent {
public:
  explicit DoubleQLearningAgent(const Hyperparameters &params,
                                std::optional<uint32_t> seed = std::nullopt)
      : TabularAgent(params, seed) {}

  void learn(const Experience &experience) override;

  /** Loads the combined table into both halves; enough to resume training. */
  void load(const std::string &filepath) override;
  void reset() override;
  std::string getName() const override { return "Double Q-Learning"; }

protected:
  /** Max of the combined estimate. learn() does not use it. */
  double bootstrapValue(const State &nextState,
                        const std::vector<Action> &validNextActions) const override {
    return qTable_.getMaxQ(nextState, validNextActions);
  }

private:
  PolicyTable tableA_{0.0};
  PolicyTable tableB_{0.0};

  /** Average of the tables that have been updated for (state, action). */
  double combined(const State &state, Action action) const;
};

} // namespace ai
} // namespace blackjack
