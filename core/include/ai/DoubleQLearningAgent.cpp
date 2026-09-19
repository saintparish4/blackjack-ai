#include "DoubleQLearningAgent.hpp"

namespace blackjack {
namespace ai {

double DoubleQLearningAgent::combined(const State &state, Action action) const {
  const uint32_t nA = tableA_.getVisitCount(state, action);
  const uint32_t nB = tableB_.getVisitCount(state, action);
  // A table that has never seen the pair still reads the 0.0 default; letting
  // it into the average would halve the other table's estimate.
  if (nA > 0 && nB > 0) {
    return 0.5 * (tableA_.get(state, action) + tableB_.get(state, action));
  }
  if (nA > 0) return tableA_.get(state, action);
  if (nB > 0) return tableB_.get(state, action);
  return 0.0;
}

void DoubleQLearningAgent::learn(const Experience &experience) {
  const State &state = experience.state;
  const Action action = experience.action;

  std::bernoulli_distribution coin(0.5);
  const bool updateA = coin(rng_);
  PolicyTable &target = updateA ? tableA_ : tableB_;
  const PolicyTable &other = updateA ? tableB_ : tableA_;

  // Select with one table, evaluate with the other.
  auto crossValue = [&](const State &next, const std::vector<Action> &valid) {
    return other.get(next, target.getMaxAction(next, valid));
  };
  double targetQ = experience.reward;
  if (!experience.done) {
    double next = crossValue(experience.nextState, nextActionsFor(experience));
    if (experience.secondNextState) {
      next += crossValue(*experience.secondNextState,
                         secondNextActionsFor(experience));
    }
    targetQ += params_.discountFactor * next;
  }

  const double currentQ = target.get(state, action);
  const double alpha = stepSize(target.getVisitCount(state, action));
  target.set(state, action, currentQ + alpha * (targetQ - currentQ));

  // Keep the acting table in step; its visit count becomes the total number
  // of updates across both halves.
  qTable_.set(state, action, combined(state, action));

  decayEpsilon();
  stepCount_++;
}

void DoubleQLearningAgent::load(const std::string &filepath) {
  TabularAgent::load(filepath);
  tableA_ = qTable_;
  tableB_ = qTable_;
}

void DoubleQLearningAgent::reset() {
  TabularAgent::reset();
  tableA_.clear();
  tableB_.clear();
}

} // namespace ai
} // namespace blackjack
