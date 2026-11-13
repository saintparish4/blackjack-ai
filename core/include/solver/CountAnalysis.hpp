#pragma once

/**
 * @file CountAnalysis.hpp
 * @brief What a card count can know, before anything tries to learn it.
 *
 * The raw material of every counting system is the effect of removal (EoR):
 * how much an expectation moves when one card of each rank leaves a full
 * shoe. A count's tags are a linear approximation to those effects, so the
 * correlation between the two says, in advance, how much a count can tell a
 * player (Griffin, The Theory of Blackjack):
 *
 *  - Betting correlation: tags against the EoR of the round's EV. How well
 *    the count ranks shoes by edge, which is all bet sizing needs.
 *  - Playing correlation, per decision: tags against the EoR of one play's
 *    gain over another. How well the count can time an index play.
 *
 * The same regression gives a linear estimate of each index: the gain moves
 * by D x beta per true count (D decks, beta the weighted regression slope of
 * EoR on tag), so it crosses zero at -gain / (D x beta). That is Griffin's
 * method, and it is an estimate of the answer key computed from the rules
 * alone, independent of any published table and of any learner.
 *
 * All EoRs are exact differences of two exact solves; nothing samples.
 */

#include "Solver.hpp"
#include <array>

namespace blackjack {
namespace solver {

/** A count's tags by value class (index 0 unused). */
using Tags = std::array<double, 11>;

/** Hi-Lo: 2-6 are +1, 7-9 are 0, tens and aces are -1. */
Tags hiLoTags();

/** EV(shoe less one card of rank r) - EV(shoe), per round, for r = 1..10. */
std::array<double, 11> effectOfRemoval(const Rules &rules, const Shoe &shoe,
                                       const Policy &policy);

/**
 * Correlation of effects with tags across the cards of `shoe` (each rank
 * weighted by how many of it there are), with the effects centred first.
 * The tags of a balanced count are already centred.
 */
double weightedCorrelation(const std::array<double, 11> &effects, const Tags &tags,
                           const Shoe &shoe);

/** The EoR analysis of one index play. */
struct IndexEstimate {
  double gain = 0.0;               ///< EV(deviation) - EV(base) off the top, after the peek
  std::array<double, 11> eor{};    ///< Change in that gain per card removed
  double slopePerTrueCount = 0.0;  ///< D x regression slope of eor on tags
  double linearIndex = 0.0;        ///< True count where the gain crosses zero
  double correlation = 0.0;        ///< Playing correlation of the count for this play
};

/**
 * Analyse the choice between `deviation` and `base` for `hand` against
 * `upCard`, with `policy` playing any later decisions.
 */
IndexEstimate estimateIndex(const Rules &rules, const Shoe &shoe, const Policy &policy,
                            const Composition &hand, int upCard, ai::Action deviation,
                            ai::Action base, const Tags &tags);

} // namespace solver
} // namespace blackjack
