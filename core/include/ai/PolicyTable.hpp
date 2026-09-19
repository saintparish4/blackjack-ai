#pragma once

#include "Agent.hpp"
#include "State.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace blackjack {
namespace ai {

/** Flat Q-table (state -> action values); state hash is direct array index.
 *  State::hash() fits in 16 bits → 65536 entries. States of agents that don't
 *  count cards all sit in the first 4096 (count bucket 0), so they touch the
 *  same memory they did before counting existed.
 *  Unvisited states return defaultValue_. */
class PolicyTable {
public:
  static constexpr size_t TABLE_SIZE = 65536;
  static constexpr size_t NUM_ACTIONS = 5;  // HIT, STAND, DOUBLE, SPLIT, SURRENDER
  using QValues = std::array<double, NUM_ACTIONS>;

  // Heap storage: 65536 rows are about 4 MB, which would overflow a stack
  // (Emscripten's is 64 KB) the moment an agent lived on one.
  explicit PolicyTable(double defaultValue = 0.0)
      : table_(TABLE_SIZE), visits_(TABLE_SIZE), visited_(TABLE_SIZE, 0),
        defaultValue_(defaultValue) {
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
      visits_[idx].fill(0);
      visited_[idx] = 1;
      ++visitedCount_;
    }
    table_[idx][static_cast<size_t>(action)] = value;
    if (visits_[idx][static_cast<size_t>(action)] < UINT32_MAX) {
      ++visits_[idx][static_cast<size_t>(action)];
    }
  }

  /** Times set() has been called for this (state, action).
   *
   *  Distinguishes "the agent learned this action is worth 0.0" from "the
   *  agent was never offered this action here, so its Q sits at the default".
   *  The two are indistinguishable by Q-value alone, and conflating them lets
   *  an untrained action win a greedy argmax against genuinely negative ones. */
  uint32_t getVisitCount(const State &state, Action action) const {
    size_t idx = state.hash();
    if (!visited_[idx]) {
      return 0;
    }
    return visits_[idx][static_cast<size_t>(action)];
  }

  bool wasActionTried(const State &state, Action action) const {
    return getVisitCount(state, action) > 0;
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

  size_t size() const { return visitedCount_; }
  bool empty() const { return visitedCount_ == 0; }

  void clear() {
    std::fill(visited_.begin(), visited_.end(), 0);
    visitedCount_ = 0;
    // table_ and visits_ entries are re-initialized lazily in set()
  }

  void saveToBinary(const std::string &filepath) const;
  void loadFromBinary(const std::string &filepath);

  /** CSV columns:
   * player_total,dealer_card,usable_ace,can_split,can_double,count_bucket,
   * Q_HIT,Q_STAND,Q_DOUBLE,Q_SPLIT,Q_SURRENDER,N_HIT,N_STAND,N_DOUBLE,N_SPLIT,N_SURRENDER
   *
   * can_split/can_double/count_bucket are part of the state key, not
   * decoration: omitting them collapses distinct Q-table slots onto duplicate
   * rows. count_bucket is 0 for agents that don't count. */
  void exportToCSV(const std::string &filepath) const;

  /** Reconstruct State from its hash index (inverse of State::hash()). */
  static State stateFromHash(size_t h) {
    State s;
    s.playerTotal   = static_cast<int>(h & 0x1F);
    s.dealerUpCard  = static_cast<int>((h >> 5) & 0x0F);
    s.hasUsableAce  = ((h >> 9) & 1) != 0;
    s.canSplit      = ((h >> 10) & 1) != 0;
    s.canDouble     = ((h >> 11) & 1) != 0;
    s.countBucket   = static_cast<int>((h >> 12) & 0x0F);
    return s;
  }

private:
  std::vector<QValues> table_;
  using VisitCounts = std::array<uint32_t, NUM_ACTIONS>;
  std::vector<VisitCounts> visits_;
  std::vector<uint8_t> visited_;
  size_t visitedCount_ = 0;
  double defaultValue_;
};
} // namespace ai
} // namespace blackjack
