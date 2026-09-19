#pragma once

#include "TabularAgent.hpp"

namespace blackjack {
namespace ai {

/** Q-learning: Q(s,a) ← Q + α[R + γ max Q(s',a') - Q]; ε-greedy
 * exploration with decay. Off-policy: it bootstraps from the greedy value of
 * the next state whatever the exploring policy actually does there. */
class QLearningAgent : public TabularAgent {
public:
  explicit QLearningAgent(const Hyperparameters &params = Hyperparameters{
                              0.1, 0.95, 1.0, 0.99995, 0.01},
                          std::optional<uint32_t> seed = std::nullopt)
      : TabularAgent(params, seed) {}

  std::string getName() const override { return "Q-Learning"; }

protected:
  double bootstrapValue(const State &nextState,
                        const std::vector<Action> &validNextActions) const override {
    return qTable_.getMaxQ(nextState, validNextActions);
  }
};

} // namespace ai
} // namespace blackjack
