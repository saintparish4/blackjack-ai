#pragma once

/**
 * @file PolicyAnalysis.hpp
 * @brief Regret: what each of a policy's decisions costs, exactly.
 *
 * Cell-match accuracy counts a mistake on hard 4 the same as one on 16 vs T,
 * and can't say what either costs. This prices them. For a policy pi and a
 * reference rho, the performance-difference identity gives
 *
 *     EV(rho) - EV(pi) = sum over decisions n reached by pi of
 *                        reach_pi(n) * [Q_rho(n, rho(n)) - Q_rho(n, pi(n))]
 *
 * where Q_rho is the solver's exact value of an action with rho playing on.
 * Grouping n by the agent's own State gives an itemised bill whose lines sum
 * to the EV gap exactly, which every report checks.
 *
 * Two references matter. Composition-dependent optimal play (the best any
 * player can do) and the best *total-dependent* policy: the best a player
 * that sees only the agent's State can do. The gap between those two is the
 * ceiling of the representation; whatever the agent loses beyond it is
 * learning error.
 */

#include "Solver.hpp"
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace blackjack {
namespace solver {

/** The agent's view of a decision point. */
ai::State stateOf(const Decision &d);

/**
 * A total-dependent policy: one action per ai::State, the agent's state space.
 * States without an entry, or whose entry isn't legal, defer to a fallback.
 */
class TabularPolicy : public Policy {
public:
  explicit TabularPolicy(const Policy &fallback) : fallback_(&fallback) {}
  void set(const ai::State &state, ai::Action action) { table_[state.hash()] = action; }
  std::optional<ai::Action> get(const ai::State &state) const;
  size_t size() const { return table_.size(); }
  bool operator==(const TabularPolicy &o) const { return table_ == o.table_; }

  ai::Action choose(const HandInfo &info, const ActionValues *unsplit) const override;
  bool needsUnsplitValues() const override { return fallback_->needsUnsplitValues(); }

private:
  std::unordered_map<size_t, ai::Action> table_;
  const Policy *fallback_;
};

/** Result of searching for the best total-dependent policy. */
struct TotalDependentSearch {
  TabularPolicy policy;
  double ev;
  std::vector<double> evByIteration; ///< evByIteration[0] is the starting policy
};

/**
 * Policy iteration over the agent's states, starting from `start`: score
 * every action at every State by its reach-weighted exact value under the
 * current policy, switch each State to its best action, re-solve, repeat
 * until nothing changes or EV stops rising. States share actions across
 * compositions, so this finds a local optimum, not a certified global one:
 * its EV is a lower bound on the representation's ceiling.
 */
TotalDependentSearch bestTotalDependentPolicy(const Rules &rules, const Shoe &shoe,
                                              const Policy &start, int maxIterations = 20);

/** One State's line on the bill. */
struct StateLoss {
  ai::State state;
  double reach = 0.0;           ///< Probability per round of deciding here
  double loss = 0.0;            ///< EV per round lost here against the reference
  ai::Action played = ai::Action::STAND;
  ai::Action reference = ai::Action::STAND; ///< The reference's play, by reach
};

struct RegretReport {
  double policyEv = 0.0;
  double referenceEv = 0.0;
  double itemisedLoss = 0.0;    ///< Sum of byState; equals referenceEv - policyEv
  std::vector<StateLoss> byState; ///< Largest loss first
};

/** Itemise reference.EV - policy.EV by the policy's States. */
RegretReport regretByState(Solver &policy, Solver &reference);

/** "H17 vs 10" style label for a State. */
std::string describe(const ai::State &s);

/**
 * Write a total-dependent policy in the format tools/farmer/policy_ev.cpp
 * reads: one line per state, "total soft up double split surrender action",
 * with Farmer's action codes (1 stand, 2 hit, 3 double, 4 split, 5 surrender).
 * Every state the engine can present is written, so Farmer never falls back
 * to its own default for one.
 */
void writeFarmerPolicy(const Policy &policy, const Rules &rules, const std::string &path);

} // namespace solver
} // namespace blackjack
