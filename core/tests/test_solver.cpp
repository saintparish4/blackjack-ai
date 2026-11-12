// Exact solver against references outside this repository.
//
// Every golden number below was produced by code nobody on this project
// wrote: Eric Farmer's combinatorial analyzer (github.com/possibly-wrong/
// blackjack at a1f7dbb, built and run by tools/farmer/run_farmer.sh) and the
// house-edge table embedded in the Wizard of Odds calculator page
// (wizardofodds.com/games/blackjack/calculator/, "optimal results" column).
// Farmer prints EVs in percent to nine decimals, so agreement is asserted to
// half of the last printed digit. If one of these fails, the solver is wrong
// or a rule is modelled differently; widening a tolerance is never the fix.

#include "solver/InfiniteDeck.hpp"
#include "solver/Solver.hpp"
#include "training/Evaluator.hpp"
#include <gtest/gtest.h>

#include <cmath>

using namespace blackjack;
using namespace blackjack::solver;
using ai::Action;

namespace {

constexpr double kFarmerTolerance = 5e-12; // 0.5e-9 percent, as a fraction

Rules farmerRules(int decks, bool h17, bool das, bool surrender) {
  Rules r;
  r.decks = decks;
  r.hitsSoft17 = h17;
  r.doubleAfterSplit = das;
  r.surrender = surrender;
  r.splitIdenticalRanksOnly = false; // Farmer splits any two tens
  return r;
}

/// One solve of Farmer's six-deck S17 rules, shared by every test that only
/// reads from it: a solve is about a second optimised and far longer under
/// the sanitizers CI runs.
Solver &sixDeckFarmer() {
  static OptimalPolicy optimal;
  static Solver solver(farmerRules(6, false, false, false), optimal);
  return solver;
}

Composition hand(std::initializer_list<int> cards) {
  Composition h;
  for (int c : cards) h.add(c);
  return h;
}

} // namespace

TEST(DealerTest, OutcomesSumToOneForEveryUpCard) {
  for (int decks : {1, 2, 6, 8}) {
    for (bool h17 : {false, true}) {
      const auto d = dealerOutcomesByUpCard(Shoe::decks(decks), h17);
      for (int up = 1; up <= 10; ++up) {
        EXPECT_NEAR(d[up].sum(), 1.0, 1e-12) << decks << " decks, up " << up;
      }
    }
  }
}

TEST(DealerTest, OnlyAceAndTenUpCardsHaveNaturals) {
  const auto d = dealerOutcomesByUpCard(Shoe::decks(6), false);
  for (int up = 2; up <= 9; ++up) EXPECT_EQ(d[up].blackjack, 0.0);
  EXPECT_NEAR(d[ACE].blackjack, 96.0 / 311.0, 1e-15); // 96 tens in 311 cards
  EXPECT_NEAR(d[TEN].blackjack, 24.0 / 311.0, 1e-15); // 24 aces
}

TEST(DealerTest, SixDeckS17MatchesFarmerTable) {
  // Farmer's "Probability of outcome of dealer's hand", 6 decks S17 (5 dp).
  struct Row { int up; double bust, p17, p18, p19, p20, p21, bj; };
  const Row rows[] = {
      {2, 0.35350, 0.13966, 0.13439, 0.13002, 0.12402, 0.11841, 0.00000},
      {6, 0.42284, 0.16571, 0.10619, 0.10643, 0.10155, 0.09728, 0.00000},
      {7, 0.26194, 0.36921, 0.13793, 0.07843, 0.07868, 0.07382, 0.00000},
      {10, 0.21247, 0.11191, 0.11167, 0.11194, 0.34001, 0.03482, 0.07717},
      {1, 0.11547, 0.13002, 0.13082, 0.13059, 0.13091, 0.05350, 0.30868},
  };
  const auto d = dealerOutcomesByUpCard(Shoe::decks(6), false);
  for (const Row &r : rows) {
    const DealerDist &x = d[r.up];
    EXPECT_NEAR(x.bust, r.bust, 5e-6) << "up " << r.up;
    EXPECT_NEAR(x.total[0], r.p17, 5e-6) << "up " << r.up;
    EXPECT_NEAR(x.total[1], r.p18, 5e-6) << "up " << r.up;
    EXPECT_NEAR(x.total[2], r.p19, 5e-6) << "up " << r.up;
    EXPECT_NEAR(x.total[3], r.p20, 5e-6) << "up " << r.up;
    EXPECT_NEAR(x.total[4], r.p21, 5e-6) << "up " << r.up;
    EXPECT_NEAR(x.blackjack, r.bj, 5e-6) << "up " << r.up;
  }
}

TEST(SolverTest, OptimalEvMatchesFarmerAndWizardOfOddsOnFiveRuleSets) {
  struct Case { int decks; bool h17, das, surrender; double farmerPercent; };
  // Wizard of Odds' table agrees with every one of these to its six
  // published decimals (0.578794, 0.668951, 0.168767, 0.485748, 0.531384).
  const Case cases[] = {
      {6, false, false, false, -0.578793657}, // the engine's Vegas Strip rules
      {6, true, true, false, -0.668950590},
      {1, true, false, false, -0.168766712},
      {2, true, false, true, -0.485748350},   // the engine's Downtown rules
      {8, false, false, true, -0.531383781},  // the engine's Atlantic City rules
  };
  OptimalPolicy optimal;
  for (const Case &c : cases) {
    Solver solver(farmerRules(c.decks, c.h17, c.das, c.surrender), optimal);
    EXPECT_NEAR(solver.expectedValue(), c.farmerPercent / 100.0, kFarmerTolerance)
        << c.decks << " decks, h17=" << c.h17 << " das=" << c.das
        << " ls=" << c.surrender;
  }
}

TEST(SolverTest, EvByUpCardMatchesFarmer) {
  const double farmer[] = {0, -34.173650853, 9.066480185, 12.327491231,
                           15.881465953, 19.839404954, 23.024228684,
                           14.257451822, 5.690672706, -4.145061167,
                           -17.323200265};
  Solver &solver = sixDeckFarmer();
  for (int up = 1; up <= 10; ++up) {
    EXPECT_NEAR(solver.expectedValue(up), farmer[up] / 100.0, kFarmerTolerance)
        << "up " << up;
  }
}

TEST(SolverTest, PerHandValuesMatchFarmer) {
  // Farmer's interactive "Enter dealer up card and player hand" output,
  // 6 decks S17 NDAS, conditioned on no dealer natural.
  struct Case { int up; Composition hand; Action action; double percent; };
  const Case cases[] = {
      {10, hand({10, 6}), Action::STAND, -54.095443902},
      {10, hand({10, 6}), Action::HIT, -53.467556245},
      {10, hand({9, 7}), Action::HIT, -53.539184207},
      {10, hand({10, 4, 2}), Action::STAND, -54.118901340},
      {10, hand({10, 4, 2}), Action::HIT, -54.132284441},
      {6, hand({6, 5}), Action::DOUBLE, 68.266469579},
      {ACE, hand({6, 5}), Action::HIT, 14.759567369},
      {ACE, hand({6, 5}), Action::DOUBLE, 12.970987826},
      {10, hand({8, 8}), Action::SPLIT, -49.254177543},
      {9, hand({9, 9}), Action::SPLIT, -10.562709003},
      {ACE, hand({1, 1}), Action::SPLIT, 12.714379565},
      {5, hand({2, 2}), Action::SPLIT, 4.305549273},
      {6, hand({1, 4}), Action::DOUBLE, 18.291740674},
  };
  Solver &solver = sixDeckFarmer();
  for (const Case &c : cases) {
    const ActionValues v = solver.peekedValues(c.hand, c.up);
    ASSERT_TRUE(v.legal(c.action));
    EXPECT_NEAR(v[c.action], c.percent / 100.0, kFarmerTolerance)
        << ai::actionToString(c.action) << " vs " << c.up;
  }
}

TEST(SolverTest, CompositionMattersWhereFarmerSaysItDoes) {
  // Farmer: "Hard 16 vs. T: stand except 88, 79, 6T, 466, 367, 33T, ..." in
  // 6 decks S17. So 10,6 hits, while 10,4,2 (not on the list) stands.
  Solver &solver = sixDeckFarmer();
  const ActionValues twoCard = solver.handValues(hand({10, 6}), 10);
  const ActionValues threeCard = solver.handValues(hand({10, 4, 2}), 10);
  EXPECT_GT(twoCard[Action::HIT], twoCard[Action::STAND]);
  EXPECT_GT(threeCard[Action::STAND], threeCard[Action::HIT]);
}

TEST(SolverTest, OptimalNeverSplitsTensSoTheTenRankRuleCostsNothing) {
  OptimalPolicy optimal;
  Rules identical = farmerRules(6, false, false, false);
  identical.splitIdenticalRanksOnly = true;
  Solver a(identical, optimal);
  EXPECT_EQ(a.expectedValue(), sixDeckFarmer().expectedValue());
}

TEST(SolverTest, BookChartIsBelowOptimalByAboutWhatWizardOfOddsSays) {
  // Wizard of Odds puts total-dependent basic strategy 0.0031% behind
  // composition-dependent play in six decks (its 0.0231 - 0.020 adjustment).
  GameRules game = GameRules::vegasStrip();
  training::BasicStrategyAgent bookAgent(game);
  AgentPolicy book(bookAgent);
  OptimalPolicy optimal;
  Solver bookSolver(Rules::fromGame(game), book);
  Solver optimalSolver(Rules::fromGame(game), optimal);
  const double gap = optimalSolver.expectedValue() - bookSolver.expectedValue();
  EXPECT_GT(gap, 0.0);
  EXPECT_NEAR(gap, 0.000031, 0.000005);
}

TEST(InfiniteDeckTest, SolverConvergesAsOneOverDecks) {
  // EV(D) = EV(inf) + c/D + O(1/D^2): the gap times D must settle.
  Rules r = farmerRules(6, false, false, false);
  const double inf = infiniteDeckExpectedValue(r);
  OptimalPolicy optimal;
  double scaled[2];
  int i = 0;
  for (int decks : {256, 1024}) {
    r.decks = decks;
    Solver solver(r, optimal);
    scaled[i++] = (solver.expectedValue() - inf) * decks;
  }
  EXPECT_GT(scaled[0], 0.0); // fewer decks favour the player
  EXPECT_NEAR(scaled[1], scaled[0], 1e-4);
}

TEST(InfiniteDeckTest, DealerMatchesAVeryLargeShoe) {
  const auto inf = infiniteDeckDealer(false);
  const auto big = dealerOutcomesByUpCard(Shoe::decks(4096), false);
  for (int up = 1; up <= 10; ++up) {
    EXPECT_NEAR(big[up].bust, inf[up].bust, 1e-5) << "up " << up;
    EXPECT_NEAR(big[up].blackjack, inf[up].blackjack, 1e-5) << "up " << up;
  }
}