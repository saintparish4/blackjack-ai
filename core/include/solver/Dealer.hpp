#pragma once

/**
 * @file Dealer.hpp
 * @brief Exact probability of each final dealer hand, by upcard.
 *
 * Every number the solver produces rests on this distribution, so it is
 * computed by brute force over every ordered sequence of dealer draws, with
 * the shoe depleted card by card. No approximation, no generated tables.
 *
 * The distribution is *unconditional*: it includes the chance the dealer has
 * blackjack. Conditioning on the peek happens downstream (Solver adds back
 * the bet a double or split doesn't lose to a dealer natural), which is the
 * convention Farmer's CA uses and the reason the two can agree to the last
 * digit.
 */

#include "Shoe.hpp"
#include <array>

namespace blackjack {
namespace solver {

/** Final dealer hand probabilities for one upcard. */
struct DealerDist {
  std::array<double, 5> total{}; ///< total[t - 17] for t = 17..21, natural excluded
  double bust = 0.0;
  double blackjack = 0.0;

  double sum() const {
    double s = bust + blackjack;
    for (double p : total) s += p;
    return s;
  }
};

/**
 * Dealer outcomes given `upCard`, drawing the hole card and every hit from
 * `shoe` (which must already exclude the upcard). The shoe is restored before
 * returning.
 */
DealerDist dealerOutcomes(Shoe &shoe, int upCard, bool hitsSoft17);

/**
 * Dealer outcomes for every upcard, each drawn from `shoe` and then played
 * out from what is left. Entry 0 is unused; an upcard with no cards in the
 * shoe gets an all-zero distribution.
 */
std::array<DealerDist, 11> dealerOutcomesByUpCard(const Shoe &shoe, bool hitsSoft17);

} // namespace solver
} // namespace blackjack