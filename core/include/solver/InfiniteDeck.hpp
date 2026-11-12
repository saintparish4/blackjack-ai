#pragma once

/**
 * @file InfiniteDeck.hpp
 * @brief The infinite-deck approximation, as an independent check on Solver.
 *
 * With infinitely many decks no draw changes the odds of the next: every
 * card is an ace with probability 1/13 and a ten with 4/13, forever. That
 * collapses a hand to its total and softness, so this file is a second,
 * deliberately different implementation of the same game: no compositions,
 * no shoe, no memo keyed by cards. Solver on a very large shoe has to
 * converge to it, and a bug in either one's dealer rules, peek handling,
 * doubling or splitting shows up as a gap that doesn't close.
 *
 * Total-dependent and composition-dependent play coincide here, so the
 * optimal policy needs no Policy object.
 */

#include "Dealer.hpp"
#include "Solver.hpp"
#include <array>

namespace blackjack {
namespace solver {

/** Dealer outcomes by upcard with constant card probabilities. */
std::array<DealerDist, 11> infiniteDeckDealer(bool hitsSoft17);

/** EV per round of optimal play (rules.decks is ignored). */
double infiniteDeckExpectedValue(const Rules &rules);

} // namespace solver
} // namespace blackjack