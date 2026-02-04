#pragma once

#include "Agent.hpp"
#include "State.hpp"
#include <array>
#include <string>
#include <unordered_map>

namespace blackjack {
namespace ai {
/**
 * @brief Stores Q-values for all state-action pairs
 *
 * Uses std::unordered_map for sparse storage (only visited states stored).
 * Alternative: flat array for dense storage if all states visited.
 */
class PolicyTable {
public:
  static constexpr size_t NUM_ACTIONS = 4;
  using QValues = std::array<double, NUM_ACTIONS>;

  /**
   * @brief Construct empty policy table with default Q-values
   */
  PolicyTable(double defaultValue = 0.0) : defaultValue_(defaultValue) {}

  /**
   * @brief Get Q-value for state-action pair
   *
   * Returns default value if state never visited.
   */
  double get(const State &state, Action action) const {
    auto it = table_.find(state);
    if (it == table_.end()) {
      return defaultValue_;
    }
    return it->second[static_cast<size_t>(action)];
  }

  /**
   * @brief Set Q-value for state-action pair
   */
  void set(const State &state, Action action, double value) {
    table_[state][static_cast<size_t>(action)] = value;
  }

  /**
   * @brief Get all Q-values for a state
   *
   * Returns array [Q(s,HIT), Q(s,STAND), Q(s,DOUBLE), Q(s,SPLIT)]
   */
  QValues getAll(const State &state) const {
    auto it = table_.find(state);
    if (it == table_.end()) {
      QValues values;
      values.fill(defaultValue_);
      return values;
    }
    return it->second;
  }

  /**
   * @brief Find action with maximum Q-value for state
   *
   * @param state The state
   * @param validActions Only consider these actions
   * @return Action with highest Q-value
   */
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

  /**
   * @brief Get maximum Q-value for state
   */
  double getMaxQ(const State &state,
                 const std::vector<Action> &validActions) const {
    double maxQ = std::numeric_limits<double>::lowest();

    for (Action action : validActions) {
      double q = get(state, action);
      maxQ = std::max(maxQ, q);
    }

    return maxQ;
  }

  /**
   * @brief Get number of states in table
   */
  size_t size() const { return table_.size(); }

  /**
   * @brief Check if table is empty
   */
  bool empty() const { return table_.empty(); }

  /**
   * @brief Clear all entries
   */
  void clear() { table_.clear(); }

  /**
   * @brief Save Q-table to binary file
   */
  void saveToBinary(const std::string &filepath) const;

  /**
   * @brief Load Q-table from binary file
   */
  void loadFromBinary(const std::string &filepath);

  /**
   * @brief Export Q-table to CSV for analysis
   *
   * Format: player_total,dealer_card,usable_ace,Q_HIT,Q_STAND,Q_DOUBLE,Q_SPLIT
   */
  void exportToCSV(const std::string &filepath) const;

  /**
   * @brief Get const iterator to table (for analysis)
   */
  auto begin() const { return table_.begin(); }
  auto end() const { return table_.end(); }

private:
  std::unordered_map<State, QValues> table_;
  double defaultValue_;
};
} // namespace ai
} // namespace blackjack