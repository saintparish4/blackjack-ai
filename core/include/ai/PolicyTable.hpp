#pragma once

#include "Agent.hpp"
#include "State.hpp"
#include <array>
#include <limits>
#include <string>
#include <unordered_map>

namespace blackjack {
namespace ai {

/** Sparse Q-table (state -> action values); unvisited states return default. */
class PolicyTable {
public:
  static constexpr size_t NUM_ACTIONS = 4;
  using QValues = std::array<double, NUM_ACTIONS>;

  PolicyTable(double defaultValue = 0.0) : defaultValue_(defaultValue) {}

  /** Returns defaultValue_ if state not in table. */
  double get(const State &state, Action action) const {
    auto it = table_.find(state);
    if (it == table_.end()) {
      return defaultValue_;
    }
    return it->second[static_cast<size_t>(action)];
  }

  void set(const State &state, Action action, double value) {
    table_[state][static_cast<size_t>(action)] = value;
  }

  /** Order: HIT, STAND, DOUBLE, SPLIT. Unvisited state returns all default. */
  QValues getAll(const State &state) const {
    auto it = table_.find(state);
    if (it == table_.end()) {
      QValues values;
      values.fill(defaultValue_);
      return values;
    }
    return it->second;
  }

  Action getMaxAction(const State &state,
                      const std::vector<Action> &validActions) const {
    double maxQ = std::numeric_limits<double>::lowest();
    Action bestAction = validActions[0];

    for (Action action : validActions) {
      double q = get(state, action);
      if (q > maxQ) {
        maxQ = q;
        bestAction = action;
      }
    }

    return bestAction;
  }

  double getMaxQ(const State &state,
                 const std::vector<Action> &validActions) const {
    double maxQ = std::numeric_limits<double>::lowest();

    for (Action action : validActions) {
      double q = get(state, action);
      maxQ = std::max(maxQ, q);
    }

    return maxQ;
  }

  size_t size() const { return table_.size(); }
  bool empty() const { return table_.empty(); }
  void clear() { table_.clear(); }

  void saveToBinary(const std::string &filepath) const;
  void loadFromBinary(const std::string &filepath);

  /** CSV columns:
   * player_total,dealer_card,usable_ace,Q_HIT,Q_STAND,Q_DOUBLE,Q_SPLIT */
  void exportToCSV(const std::string &filepath) const;

  auto begin() const { return table_.begin(); }
  auto end() const { return table_.end(); }

private:
  std::unordered_map<State, QValues> table_;
  double defaultValue_;
};
} // namespace ai
} // namespace blackjack