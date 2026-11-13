// Effects of removal and the indices they predict, against published values.
//
// Griffin's single-deck effects of removal (The Theory of Blackjack, as
// reproduced by Wizard of Odds and most counting texts) and Hi-Lo's
// betting correlation of 0.97 are the external references. His figures
// assume slightly different rules from this engine's, so agreement is
// asserted to 0.06% per card, about a tenth of the largest effect.

#include "solver/CountAnalysis.hpp"
#include "training/Evaluator.hpp"
#include <gtest/gtest.h>

#include <cmath>

using namespace blackjack;
using namespace blackjack::solver;
using ai::Action;

TEST(CountAnalysisTest, HiLoTagsAreBalanced) {
  const Tags t = hiLoTags();
  const Shoe deck = Shoe::decks(1);
  double sum = 0.0;
  for (int r = 1; r <= 10; ++r) sum += deck.n[r] * t[r];
  EXPECT_EQ(sum, 0.0);
}

TEST(CountAnalysisTest, SingleDeckEffectsOfRemovalAreGriffins) {
  // Percent of a bet per card removed: A, 2, 3, ..., 9, T.
  const double griffin[] = {0, -0.61, 0.38, 0.44, 0.55, 0.69, 0.46, 0.28, 0.00, -0.18, -0.51};
  Rules r;
  r.decks = 1;
  OptimalPolicy optimal;
  const auto e = effectOfRemoval(r, Shoe::decks(1), optimal);
  for (int rank = 1; rank <= 10; ++rank) {
    EXPECT_NEAR(e[rank] * 100, griffin[rank], 0.06) << "rank " << rank;
  }
  // To first order a random card changes nothing, so the card-weighted mean
  // effect is near zero. Not exactly zero, and not by rounding: optimal play
  // re-optimised for a known 51-card deck is worth a little more on average
  // (playing on information is convex), here +0.025% per card.
  double weighted = 0.0;
  for (int rank = 1; rank <= 10; ++rank) weighted += Shoe::decks(1).n[rank] * e[rank];
  EXPECT_GT(weighted / 52.0, 0.0);
  EXPECT_LT(weighted / 52.0, 5e-4);
}

TEST(CountAnalysisTest, HiLoBettingCorrelationIsThePublishedPointNineSeven) {
  Rules r;
  r.decks = 1;
  OptimalPolicy optimal;
  const auto e = effectOfRemoval(r, Shoe::decks(1), optimal);
  const double bc = weightedCorrelation(e, hiLoTags(), Shoe::decks(1));
  EXPECT_GT(bc, 0.955);
  EXPECT_LT(bc, 0.975);
}

TEST(CountAnalysisTest, LinearIndicesLandWithinOneOfEveryIllustrious18Index) {
  const GameRules game = GameRules::vegasStrip();
  training::BasicStrategyAgent bookAgent(game);
  AgentPolicy book(bookAgent);
  const Rules rules = Rules::fromGame(game);
  for (const training::IndexPlay &p : training::illustrious18()) {
    Composition h;
    if (p.pair) { h.add(10); h.add(10); }
    else if (p.total >= 12) { h.add(10); h.add(p.total - 10); }
    else { h.add(p.total - 2); h.add(2); }
    const IndexEstimate est = estimateIndex(rules, Shoe::decks(6), book, h, p.dealer,
                                            p.atOrAbove, p.below, hiLoTags());
    EXPECT_GT(est.slopePerTrueCount, 0.0) << p.total << " v " << p.dealer;
    EXPECT_GT(est.correlation, 0.5) << p.total << " v " << p.dealer;
    EXPECT_NEAR(est.linearIndex, p.index, 1.0) << p.total << " v " << p.dealer;
  }
}
