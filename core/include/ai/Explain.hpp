#pragma once

#include "Agent.hpp"
#include "State.hpp"
#include <string>

namespace blackjack {
namespace ai {

/**
 * Plain-English reasons and confidence labels for a recommended action.
 *
 * Shared by the terminal coach (play --beginner) and the browser demo, so the
 * two never explain the same decision differently.
 */

/** Confidence from the Q-value margin between the best and second-best action. */
inline const char *confidenceLabel(double margin) {
  if (margin > 0.15) return "High";
  if (margin > 0.05) return "Medium";
  return "Low";
}

/** Why a player would take this action in this state, in one sentence. */
inline std::string actionReason(Action action, const State &state) {
  const int dealer = (state.dealerUpCard == 1) ? 11 : state.dealerUpCard;
  const int player = state.playerTotal;

  if (action == Action::SURRENDER) {
    return "Your odds are very poor here, so cutting your losses is the right call.";
  }
  if (action == Action::DOUBLE) {
    if (state.hasUsableAce)
      return "A soft hand can't bust on one card, and the dealer's weak card makes this a spot to press.";
    if (player == 11) return "11 is a great doubling spot: most cards give you 21 or close.";
    if (player == 10) return "10 is strong for doubling: you're likely to land near 20.";
    if (player == 9 && dealer >= 3 && dealer <= 6)
      return "9 against a weak dealer card is a good chance to double and profit.";
    return "The math favors risking more here, so doubling is the right move.";
  }
  if (action == Action::SPLIT) {
    if (state.hasUsableAce && player == 12)
      return "Two aces are one soft 12 but two strong starting hands.";
    if (player == 16) return "16 is the worst total in the game; two hands starting on 8 are far better.";
    return "Splitting turns a weak hand into two chances to win.";
  }
  if (action == Action::STAND) {
    if (player >= 17)
      return "You're in solid territory: any new card risks busting you.";
    if (dealer >= 2 && dealer <= 6)
      return "The dealer's weak card means they'll often bust on their own, so there's no need to risk it.";
    return "Standing gives the dealer a chance to bust; the math favors it here.";
  }
  // HIT
  if (player <= 11 && !state.hasUsableAce) {
    return "You can't bust from here, so it's always safe to take another card.";
  }
  if (state.hasUsableAce && player <= 17) {
    return "A soft hand can't bust on the next card, so improving it is free.";
  }
  if (dealer >= 7) {
    return "The dealer's strong card means you need more to compete, so take a card.";
  }
  return "Taking another card is the best play in this spot.";
}

} // namespace ai
} // namespace blackjack
