#pragma once

#include "TabularAgent.hpp"

namespace blackjack {
namespace ai {

/**
 * Expected SARSA: Q(s,a) ← Q + α[R + γ Σ π(a'|s') Q(s',a') - Q].
 *
 * Bootstraps from the expected value of the next state under the agent's own
 * ε-greedy policy instead of its maximum. Averaging over the next action
 * removes the variance of sampling it and the upward bias of taking a max over
 * noisy estimates; the price is that it learns the value of the exploring
 * policy, which is why ε keeps a floor rather than decaying to zero.
 */
class ExpectedSarsaAgent : public TabularAgent {
public:
  explicit ExpectedSarsaAgent(const Hyperparameters &params,
                              std::optional<uint32_t> seed = std::nullopt)
      : TabularAgent(params, seed) {}

  std::string getName() const override { return "Expected SARSA"; }

protected:
  double bootstrapValue(const State &nextState,
                        const std::vector<Action> &validNextActions) const override;
};

} // namespace ai
} // namespace blackjack
