// The engine against exact values from the solver.
//
// Sampling tests, sized like the rest of the suite: 4.5 standard errors puts
// a spurious failure near 1 in 150,000 per line, and the seed is fixed, so a
// pass stays a pass. A real disagreement between engine and solver (a rule
// played differently, a payout mis-credited) moves one of these lines by far
// more than that once the sample is large: scripts/audit.cpp is the large
// version; experiment 003 records its 20M-round run.

#include "solver/Solver.hpp"
#include "training/EngineAudit.hpp"
#include "training/RunConfig.hpp"
#include <gtest/gtest.h>

#include <cmath>

using namespace blackjack;
using namespace blackjack::training;

namespace {

void expectAllLinesAgree(const GameRules &rules, uint64_t rounds) {
  AuditOptions options;
  options.rounds = rounds;
  options.seed = 7;
  for (const AuditLine &l : auditEngine(rules, options)) {
    EXPECT_LT(std::fabs(l.z()), 4.5)
        << l.name << ": exact " << l.exact << ", engine " << l.mean << " +/- " << l.stdErr;
    EXPECT_GT(l.samples, 0u) << l.name;
  }
}

} // namespace

TEST(EngineAuditTest, ExactNaturalProbabilityIsTheTextbookValue) {
  // 2 x (24/312) x (96/311) in six decks: 4.749%.
  AuditOptions options;
  options.rounds = 1000;
  const auto lines = auditEngine(GameRules::vegasStrip(), options);
  ASSERT_EQ(lines[0].name, "P(player natural)");
  EXPECT_NEAR(lines[0].exact, 2.0 * 24 * 96 / (312.0 * 311.0), 1e-15);
  EXPECT_NEAR(lines[0].exact, 0.04749, 5e-6);
}

TEST(EngineAuditTest, ExactBookEvIsTheSolversValue) {
  AuditOptions options;
  options.rounds = 1000;
  const auto lines = auditEngine(GameRules::vegasStrip(), options);
  for (const auto &l : lines) {
    if (l.name == "EV per round, book") {
      EXPECT_NEAR(l.exact, -0.00581852995, 5e-12);
      return;
    }
  }
  FAIL() << "no book EV line";
}

TEST(EngineAuditTest, ZeroSpreadLineFailsUnlessItMatchesExactly) {
  // A constant tally has no standard error. Matching the exact value is a
  // pass; missing it (the engine never plays an opening the solver says
  // happens) must not hide behind a z of 0.
  EXPECT_EQ((AuditLine{"same", -0.5, -0.5, 0.0, 100}.z()), 0.0);
  EXPECT_TRUE(std::isinf(AuditLine{"never seen", 0.02, 0.0, 0.0, 100}.z()));
  EXPECT_TRUE(std::isinf(AuditLine{"no samples", 0.3, 0.0, 0.0, 0}.z()));
  EXPECT_GT((AuditLine{"seen, not expected", 0.0, 0.01, 0.0, 100}.z()), 4.5);
}

TEST(EngineAuditTest, VegasStripEngineMatchesSolver) {
  expectAllLinesAgree(GameRules::vegasStrip(), 300'000);
}

TEST(EngineAuditTest, SingleDeckEngineMatchesSolver) {
  expectAllLinesAgree(GameRules::singleDeck(), 300'000);
}

TEST(EngineAuditTest, DowntownEngineMatchesSolverWithSurrender) {
  expectAllLinesAgree(GameRules::downtown(), 300'000);
}

TEST(GameRulesTest, EveryPresetDiffersFromEveryOther) {
  // The european preset was vegasStrip under another name.
  const char *names[] = {"vegas-strip", "downtown", "atlantic-city", "single-deck"};
  auto key = [](const GameRules &r) {
    return std::make_tuple(r.numDecks, r.dealerHitsSoft17, r.blackjackPayout,
                           r.surrender, r.penetration);
  };
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = i + 1; j < 4; ++j)
      EXPECT_NE(key(rulesFromPreset(names[i])), key(rulesFromPreset(names[j])))
          << names[i] << " vs " << names[j];
}

TEST(GameRulesTest, RemovedPresetIsRejected) {
  EXPECT_THROW(rulesFromPreset("european"), std::invalid_argument);
}
