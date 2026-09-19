#include "ConvergenceReport.hpp"
#include "../ai/Agent.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace blackjack {
namespace training {

// ---- construction ----

ConvergenceReport::ConvergenceReport(double passingThreshold,
                                     size_t maxDivergencesShown)
    : passingThreshold_(passingThreshold),
      maxDivergencesShown_(maxDivergencesShown) {}

// ---- public interface ----

ConvergenceResult ConvergenceReport::analyze(ai::Agent& agent,
                                             const BasicStrategy& basicStrategy,
                                             bool allowSurrender) const {
    ConvergenceResult result;

    // buildAuditStates only reads GameRules::surrender; the rest of the rules
    // do not change which states or actions the audit enumerates.
    GameRules rules;
    rules.surrender = allowSurrender;

    for (HandContext context : kAllHandContexts) {
        ContextAudit& slot = context == HandContext::TwoCard   ? result.twoCard
                           : context == HandContext::MultiCard ? result.multiCard
                                                               : result.pairs;

        for (const AuditState& entry : buildAuditStates(rules, context)) {
            // Skip states where some legal action was never trained: its Q is
            // still the initial 0.0, which beats every genuinely negative
            // alternative, so the argmax is an initialisation artefact rather
            // than a learned decision.
            const ai::State asked = auditView(agent, entry.state);
            bool trained = true;
            for (ai::Action action : entry.validActions) {
                if (agent.getVisitCount(asked, action) == 0) {
                    trained = false;
                    break;
                }
            }
            if (!trained) {
                ++slot.untrained;
                continue;
            }

            ++slot.scored;
            ai::Action agentAction =
                agent.chooseAction(asked, entry.validActions, false);

            if (basicStrategy.isCorrectAction(entry.state, agentAction, allowSurrender)) {
                ++slot.matches;
            } else {
                Divergence div;
                div.state         = entry.state;
                div.agentAction   = agentAction;
                // Show the play actually available: on a hand that has
                // already hit, the book's DOUBLE means its fallback.
                div.optimalAction = basicStrategy.getAction(entry.state, allowSurrender);
                if (div.optimalAction == ai::Action::DOUBLE && !entry.state.canDouble) {
                    div.optimalAction = basicStrategy.fallbackWhenCannotDouble(entry.state);
                }
                div.qMargin       = computeQMargin(agent, asked, entry.validActions);
                div.isCritical    = isCriticalState(entry.state);
                div.context       = context;
                result.divergences.push_back(div);
            }
        }
    }

    result.totalStates     = result.twoCard.scored + result.multiCard.scored +
                             result.pairs.scored;
    result.matchingStates  = result.twoCard.matches + result.multiCard.matches +
                             result.pairs.matches;
    result.untrainedStates = result.twoCard.untrained + result.multiCard.untrained +
                             result.pairs.untrained;

    result.accuracy = result.totalStates > 0
        ? static_cast<double>(result.matchingStates) / result.totalStates
        : 0.0;
    result.passed = result.accuracy >= passingThreshold_;

    // Most confident mistakes first
    std::sort(result.divergences.begin(), result.divergences.end(),
              [](const Divergence& a, const Divergence& b) {
                  // Critical states rank above minor at equal margin
                  if (a.isCritical != b.isCritical) return a.isCritical > b.isCritical;
                  return a.qMargin > b.qMargin;
              });

    return result;
}

void ConvergenceReport::print(const ConvergenceResult& result,
                              std::ostream& out) const {
    out << "\n=== Convergence Report ===\n";
    out << "Strategy accuracy : " << std::fixed << std::setprecision(1)
        << (result.accuracy * 100) << "% ("
        << result.matchingStates << "/" << result.totalStates << " states)\n";
    out << "  two-card hands  : " << std::setprecision(1)
        << (result.twoCard.accuracy() * 100) << "% ("
        << result.twoCard.matches << "/" << result.twoCard.scored
        << ")  -- doubling and surrender live here\n";
    out << "  multi-card hands: " << std::setprecision(1)
        << (result.multiCard.accuracy() * 100) << "% ("
        << result.multiCard.matches << "/" << result.multiCard.scored
        << ")  -- hit/stand only\n";
    out << "  pairs           : " << std::setprecision(1)
        << (result.pairs.accuracy() * 100) << "% ("
        << result.pairs.matches << "/" << result.pairs.scored
        << ")  -- split or play the total\n";
    out << "Threshold         : " << (passingThreshold_ * 100) << "%\n";
    out << "Status            : " << (result.passed ? "PASS ✓" : "FAIL ✗") << "\n";
    if (result.untrainedStates > 0) {
        out << "Not scored        : " << result.untrainedStates
            << " state(s) where some legal action was never trained\n";
    }

    if (result.divergences.empty()) {
        out << "No divergences from basic strategy.\n";
        out << "==========================\n";
        return;
    }

    size_t critCount = 0;
    for (const auto& d : result.divergences) {
        if (d.isCritical) ++critCount;
    }
    out << "Divergences       : " << result.divergences.size()
        << " (" << critCount << " critical, "
        << (result.divergences.size() - critCount) << " minor)\n";

    // --- Summary table (top N by margin) ---
    size_t shown = std::min(maxDivergencesShown_, result.divergences.size());
    out << "\nTop " << shown << " divergences (critical first, then by Q-value margin):\n";
    out << std::left
        << std::setw(26) << "State"
        << std::setw(12) << "Agent"
        << std::setw(12) << "Optimal"
        << std::right << std::setw(10) << "Margin"
        << std::left  << std::setw(10) << "  Type"
        << "\n";
    out << std::string(70, '-') << "\n";

    for (size_t i = 0; i < shown; ++i) {
        const Divergence& d = result.divergences[i];
        out << std::left  << std::setw(26) << describe(d)
            << std::setw(12) << ai::actionToString(d.agentAction)
            << std::setw(12) << ai::actionToString(d.optimalAction)
            << std::right << std::fixed << std::setw(9) << std::setprecision(4) << d.qMargin
            << std::left  << (d.isCritical ? "  CRITICAL" : "  minor")
            << "\n";
    }

    // --- Critical divergences listed separately for quick diagnosis ---
    if (critCount > 0) {
        out << "\nCritical divergences:\n";
        for (const auto& d : result.divergences) {
            if (!d.isCritical) continue;
            out << "  " << std::left << std::setw(24) << describe(d)
                << " agent=" << std::setw(9) << ai::actionToString(d.agentAction)
                << " optimal=" << std::setw(9) << ai::actionToString(d.optimalAction)
                << " margin=" << std::fixed << std::setprecision(4) << d.qMargin
                << "\n";
        }
    }

    out << "==========================\n";
}

// ---- private helpers ----

bool ConvergenceReport::isCriticalState(const ai::State& state) {
    // High-frequency / high-stakes decisions that have a large impact on EV.
    // These are the states that appear most often and where mistakes cost the most.
    if (state.hasUsableAce) {
        return state.playerTotal == 18; // soft 18: stand/hit dilemma vs 9-A
    }
    // Hard hands with high strategic variance
    const int p = state.playerTotal;
    const int d = (state.dealerUpCard == 1) ? 11 : state.dealerUpCard;
    if (p >= 12 && p <= 16 && d >= 7) return true;   // high bust-risk vs strong dealer
    if ((p == 10 || p == 11) && d >= 9) return true; // missed double-down opportunities
    if ((p == 15 && d == 10) || (p == 16 && (d == 9 || d == 10 || d == 11)))
        return true; // surrender states
    return false;
}

double ConvergenceReport::computeQMargin(ai::Agent& agent,
                                         const ai::State& state,
                                         const std::vector<ai::Action>& validActions) {
    if (validActions.size() < 2) return 0.0;

    double top1 = -std::numeric_limits<double>::max();
    double top2 = -std::numeric_limits<double>::max();
    for (ai::Action a : validActions) {
        double q = agent.getQValue(state, a);
        if (q > top1) { top2 = top1; top1 = q; }
        else if (q > top2) { top2 = q; }
    }
    return (top2 == -std::numeric_limits<double>::max()) ? 0.0 : (top1 - top2);
}

const char* ConvergenceReport::contextName(HandContext context) {
    switch (context) {
    case HandContext::TwoCard:   return "2card";
    case HandContext::MultiCard: return "multi";
    case HandContext::Pair:      return "pair";
    }
    return "?";
}

std::string ConvergenceReport::describe(const Divergence& d) {
    const int dc = d.state.dealerUpCard;
    const std::string dealer = (dc == 1) ? "A" : (dc == 10 ? "T" : std::to_string(dc));
    std::string hand;
    if (d.context == HandContext::Pair) {
        const int rank = d.state.hasUsableAce ? 1 : d.state.playerTotal / 2;
        hand = "pair " + std::string(rank == 1 ? "A" : rank == 10 ? "T"
                                                      : std::to_string(rank)) + "s";
    } else {
        hand = (d.state.hasUsableAce ? "soft " : "hard ") +
               std::to_string(d.state.playerTotal);
    }
    return hand + " vs " + dealer + " [" + contextName(d.context) + "]";
}

} // namespace training
} // namespace blackjack
