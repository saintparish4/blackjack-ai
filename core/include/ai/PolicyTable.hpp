#pragma once

#include "Agent.hpp"
#include "State.hpp"
#include <array>
#include <bitset>
#include <limits>
#include <string>

namespace blackjack {
namespace ai {

/** Flat Q-table (state -> action values); state hash is direct array index.
 *  State::hash() fits in 12 bits → 4096 entries max.
 *  Unvisited states return defaultValue_. */
class PolicyTable {
public:
  static constexpr size_t TABLE_SIZE = 4096;
  static constexpr size_t NUM_ACTIONS = 5;  // HIT, STAND, DOUBLE, SPLIT, SURRENDER
  using QValues = std::array<double, NUM_ACTIONS>;

  explicit PolicyTable(double defaultValue = 0.0) : defaultValue_(defaultValue) {
    for (auto &row : table_) {
      row.fill(defaultValue_);
    }
  }

  /** Returns defaultValue_ if state not visited. */
  double get(const State &state, Action action) const {
    size_t idx = state.hash();
    if (!visited_[idx]) {
      return defaultValue_;
    }
    return table_[idx][static_cast<size_t>(action)];
  }

  void set(const State &state, Action action, double value) {
    size_t idx = state.hash();
    if (!visited_[idx]) {
      table_[idx].fill(defaultValue_);
      visited_[idx] = true;
    }
    table_[idx][static_cast<size_t>(action)] = value;
  }

  /** Order: HIT, STAND, DOUBLE, SPLIT, SURRENDER. Unvisited state returns all default. */
  QValues getAll(const State &state) const {
    size_t idx = state.hash();
    if (!visited_[idx]) {
      QValues values;
      values.fill(defaultValue_);
      return values;
    }
    return table_[idx];
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
      maxQ = std::max(maxQ, get(state, action));
    }

    return maxQ;
  }

  size_t size() const { return visited_.count(); }
  bool empty() const { return visited_.none(); }

  void clear() {
    visited_.reset();
    // table_ entries are re-initialized lazily in set() after a clear
  }

  void saveToBinary(const std::string &filepath) const;
  void loadFromBinary(const std::string &filepath);

  /** CSV columns:
   * player_total,dealer_card,usable_ace,Q_HIT,Q_STAND,Q_DOUBLE,Q_SPLIT,Q_SURRENDER */
  void exportToCSV(const std::string &filepath) const;

private:
  std::array<QValues, TABLE_SIZE> table_;
  std::bitset<TABLE_SIZE> visited_;
  double defaultValue_;

  /** Reconstruct State from its hash index (inverse of State::hash()). */
  static State stateFromHash(size_t h) {
    State s;
    s.playerTotal   = static_cast<int>(h & 0x1F);
    s.dealerUpCard  = static_cast<int>((h >> 5) & 0x0F);
    s.hasUsableAce  = ((h >> 9) & 1) != 0;
    s.canSplit      = ((h >> 10) & 1) != 0;
    s.canDouble     = ((h >> 11) & 1) != 0;
    return s;
  }
};
} // namespace ai
} // namespace blackjack
