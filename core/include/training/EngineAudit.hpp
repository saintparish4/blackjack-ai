#pragma once

/**
 * @file EngineAudit.hpp
 * @brief The engine, measured against the exact solver.
 *
 * Before this, the only evidence the engine played correct blackjack was that
 * basic strategy's EV landed "in line with the published house edge". This
 * plays rounds through BlackjackGame and compares each tallied quantity with
 * its exact value from solver::Solver, as a z-score:
 *
 *  - Dealer pass: a player who stands on everything, so the dealer's draws
 *    are isolated. P(player natural), P(dealer natural) by upcard, and the
 *    dealer's bust rate by upcard (S17/H17 shows up here first).
 *  - Book pass: the repo's basic strategy chart, EV per round overall and
 *    split by the opening play (stand, hit, double, split, surrender) and by
 *    naturals. A bug in doubling or split credit moves one line, not all.
 *
 * Every round is dealt from a freshly shuffled shoe (penetration 0), because
 * that is what the solver computes. A cut card changes the answer (the
 * cut-card effect), which is why the optional cut-card line is reported
 * separately and expected to miss.
 *
 * Seeded: same rules, rounds and seed, same report.
 */

#include "../ai/Agent.hpp"
#include "../game/BlackjackGame.hpp"
#include "../game/GameRules.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace blackjack {
namespace training {

/** One audited quantity: its exact value and the engine's estimate. */
struct AuditLine {
  std::string name;
  double exact = 0.0;
  double mean = 0.0;    ///< Engine estimate
  double stdErr = 0.0;  ///< Of the estimate
  uint64_t samples = 0; ///< Rounds the estimate averages over

  /** Standard errors between engine and solver. */
  double z() const { return stdErr > 0.0 ? (mean - exact) / stdErr : 0.0; }
};

struct AuditOptions {
  uint64_t rounds = 1'000'000; ///< Per pass
  uint32_t seed = 1;
  /// Rounds of the book at the rules' own penetration. Their EV is compared
  /// with the off-the-top value, so the gap is the cut-card effect. 0 = skip.
  uint64_t cutCardRounds = 0;
};

/** Play one round greedily with `agent`, exactly as Evaluator does, and
 *  return its result in units of the initial bet (both hands of a split). */
double playOneRound(ai::Agent &agent, BlackjackGame &game);

/** Audit the engine on these rules. Lines are in a fixed order. */
std::vector<AuditLine> auditEngine(const GameRules &rules, const AuditOptions &options);

} // namespace training
} // namespace blackjack
