// Regret, reach probabilities and the best total-dependent policy.
//
// The goldens here were computed by Farmer's CA through tools/farmer/
// policy_ev, which plays an exported total-dependent policy with his own
// code: the book chart (-0.581852995%), the best total-dependent policy
// (-0.580301250%) and the Downtown book with surrender (-0.504177180%).
// The identities (reach sums, regret sums) need no golden: they hold to
// rounding error or the analysis is wrong.

#include "solver/PolicyAnalysis.hpp"
#include "training/Evaluator.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>

using namespace blackjack;
using namespace blackjack::solver;
using ai::Action;

namespace {

constexpr double kFarmerTolerance = 5e-12;

struct Book {
  GameRules game;
  training::BasicStrategyAgent agent;
  AgentPolicy policy;
  explicit Book(const GameRules &g) : game(g), agent(g), policy(agent) {}
};

} // namespace

TEST(RegretTest, BookEvMatchesFarmerPlayingTheExportedChart) {
  Book book(GameRules::vegasStrip());
  Solver s(Rules::fromGame(book.game), book.policy);
  EXPECT_NEAR(s.expectedValue(), -0.581852995 / 100, kFarmerTolerance);

  Book downtown(GameRules::downtown());
  Solver d(Rules::fromGame(downtown.game), downtown.policy);
  EXPECT_NEAR(d.expectedValue(), -0.504177180 / 100, kFarmerTolerance);
}

TEST(RegretTest, OpeningReachIsEveryNonNaturalDeal) {
  Book book(GameRules::vegasStrip());
  Solver s(Rules::fromGame(book.game), book.policy);
  double opening = 0.0;
  for (const Decision &d : s.decisions()) {
    if (d.pairCard == 0 && d.hand.cards == 2) opening += d.reach;
  }
  EXPECT_NEAR(opening, 1.0 - 2.0 * 24 * 96 / (312.0 * 311.0), 1e-14);
}

TEST(RegretTest, ItemisedRegretSumsToTheEvGapExactly) {
  Book book(GameRules::vegasStrip());
  OptimalPolicy optimal;
  const Rules rules = Rules::fromGame(book.game);
  Solver bookSolver(rules, book.policy);
  Solver optimalSolver(rules, optimal);
  const RegretReport r = regretByState(bookSolver, optimalSolver);
  EXPECT_GT(r.referenceEv, r.policyEv);
  EXPECT_NEAR(r.itemisedLoss, r.referenceEv - r.policyEv, 1e-15);
  double sum = 0.0;
  for (const StateLoss &l : r.byState) sum += l.loss;
  EXPECT_NEAR(sum, r.itemisedLoss, 1e-15);
}

TEST(RegretTest, RegretAgainstItselfIsZero) {
  Book book(GameRules::vegasStrip());
  const Rules rules = Rules::fromGame(book.game);
  Solver a(rules, book.policy);
  Solver b(rules, book.policy);
  const RegretReport r = regretByState(a, b);
  for (const StateLoss &l : r.byState) EXPECT_EQ(l.loss, 0.0) << describe(l.state);
}

TEST(RegretTest, BestTotalDependentPolicyBeatsTheBookOnlyOnMultiCardSixteenVsTen) {
  Book book(GameRules::vegasStrip());
  OptimalPolicy optimal;
  const Rules rules = Rules::fromGame(book.game);
  const TotalDependentSearch td = bestTotalDependentPolicy(rules, Shoe::decks(6), book.policy);
  Solver optimalSolver(rules, optimal);
  EXPECT_NEAR(td.ev, -0.580301250 / 100, kFarmerTolerance);
  EXPECT_LT(td.ev, optimalSolver.expectedValue()); // the representation's ceiling

  Solver bookSolver(rules, book.policy);
  Solver tdSolver(rules, td.policy);
  int changed = 0;
  for (const StateLoss &l : regretByState(bookSolver, tdSolver).byState) {
    if (std::fabs(l.loss) < 1e-12) continue;
    ++changed;
    EXPECT_EQ(l.state.playerTotal, 16);
    EXPECT_EQ(l.state.dealerUpCard, 10);
    EXPECT_FALSE(l.state.canDouble); // three or more cards
    EXPECT_EQ(l.played, Action::HIT);
    EXPECT_EQ(l.reference, Action::STAND);
  }
  EXPECT_EQ(changed, 1);
}

TEST(RegretTest, TabularPolicyDefersWhenItsEntryIsIllegal) {
  Book book(GameRules::vegasStrip());
  TabularPolicy t(book.policy);
  ai::State s(11, 6, false, false, false); // hard 11, three cards
  t.set(s, Action::DOUBLE);
  Composition h;
  h.add(5);
  h.add(4);
  h.add(2);
  const HandInfo info{h, 6, false, false, false, false};
  EXPECT_EQ(t.choose(info, nullptr), Action::HIT); // the book's play
}

TEST(RegretTest, FarmerPolicyFileCoversEveryState) {
  Book book(GameRules::vegasStrip());
  const std::string path = ::testing::TempDir() + "book.policy";
  writeFarmerPolicy(book.policy, Rules::fromGame(book.game), path);
  std::ifstream in(path);
  int lines = 0, total, soft, up, dbl, split, surr, action;
  while (in >> total >> soft >> up >> dbl >> split >> surr >> action) {
    ++lines;
    EXPECT_GE(action, 1);
    EXPECT_LE(action, 4); // no surrender on Vegas Strip
    if (!dbl) {
      EXPECT_NE(action, 3);
    }
    if (!split) {
      EXPECT_NE(action, 4);
    }
  }
  // Per upcard: 18 hard totals after a hit, 17 opening, 9 opening pairs,
  // 10 soft after a hit, 9 soft opening, and A,A.
  EXPECT_EQ(lines, 10 * (18 + 17 + 9 + 10 + 9 + 1));
}
