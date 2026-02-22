#pragma once

#include "../ai/Agent.hpp"
#include "../ai/State.hpp"
#include "Evaluator.hpp"
#include <iostream>
#include <vector>

namespace blackjack {
namespace training {

/** One state where the agent's greedy choice diverges from basic strategy. */
struct Divergence {
    ai::State   state;
    ai::Action  agentAction;   ///< What the agent chose
    ai::Action  optimalAction; ///< What basic strategy prescribes
    double      qMargin;       ///< Q-value gap between best and second-best valid action
    bool        isCritical;    ///< High-frequency / high-stakes state
};

/** Output from a full convergence analysis. */
struct ConvergenceResult {
    double               accuracy       = 0.0;   ///< Fraction of states matching basic strategy (0-1)
    bool                 passed         = false;  ///< accuracy >= passing threshold
    size_t               totalStates    = 0;
    size_t               matchingStates = 0;
    std::vector<Divergence> divergences;          ///< All divergent states, sorted by qMargin desc
};

/**
 * Exhaustive comparison of the agent's greedy policy against basic strategy.
 *
 * Iterates all valid (playerTotal 4-21) × (dealerUpCard 1-10) × (soft/hard)
 * states and records every state where the agent disagrees with BasicStrategy.
 * Divergences are ranked by Q-value margin so the most confident mistakes
 * surface first.
 *
 * Usage:
 *   ConvergenceReport report;
 *   auto result = report.analyze(agent, evaluator.getBasicStrategy());
 *   report.print(result);
 */
class ConvergenceReport {
public:
    /**
     * @param passingThreshold  Minimum accuracy fraction considered "pass" (default 0.90).
     * @param maxDivergencesShown  Max rows in the summary table (default 15).
     */
    explicit ConvergenceReport(double passingThreshold    = 0.90,
                               size_t maxDivergencesShown = 15);

    /** Run the analysis. The agent is queried in exploit mode (training=false). */
    ConvergenceResult analyze(ai::Agent& agent,
                              const BasicStrategy& basicStrategy) const;

    /** Print a formatted report to the given stream (default: stdout). */
    void print(const ConvergenceResult& result,
               std::ostream& out = std::cout) const;

private:
    double passingThreshold_;
    size_t maxDivergencesShown_;

    /** True for high-frequency / strategically critical states. */
    static bool isCriticalState(const ai::State& state);

    /**
     * Q-value margin: difference between the top and second-best Q-value
     * across valid actions. Large margin → agent is confident in its choice.
     */
    static double computeQMargin(ai::Agent& agent,
                                 const ai::State& state,
                                 const std::vector<ai::Action>& validActions);

    /** Build the valid action set for a state, matching compareWithBasicStrategy logic. */
    static std::vector<ai::Action> validActionsForState(const ai::State& state);
};

} // namespace training
} // namespace blackjack
