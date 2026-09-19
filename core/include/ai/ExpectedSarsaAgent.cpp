#include "ExpectedSarsaAgent.hpp"

namespace blackjack {
namespace ai {

double ExpectedSarsaAgent::bootstrapValue(
    const State &nextState, const std::vector<Action> &validNextActions) const {
  // ε-greedy puts ε/|A| on every legal action and the remaining 1-ε on the
  // greedy one, so the expectation is ε·mean(Q) + (1-ε)·Q(greedy).
  double sum = 0.0;
  for (Action a : validNextActions) {
    sum += qTable_.get(nextState, a);
  }
  const double mean = sum / static_cast<double>(validNextActions.size());
  const double greedy =
      qTable_.get(nextState, qTable_.getMaxAction(nextState, validNextActions));
  return epsilon_ * mean + (1.0 - epsilon_) * greedy;
}

} // namespace ai
} // namespace blackjack
