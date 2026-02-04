#pragma once

#include <cstddef>

namespace blackjack {

/** House rules and table config. */
struct GameRules {
    size_t numDecks = 6;
    bool dealerHitsSoft17 = true;
    double blackjackPayout = 1.5;  // 3:2 = 1.5, 6:5 = 1.2
    bool doubleAfterSplit = true;
    bool resplitAces = false;
    int maxSplits = 3;
    bool surrender = false;
    double penetration = 0.75;  // fraction of shoe dealt before reshuffle

    /** Returns stake + winnings (blackjack uses blackjackPayout multiplier). */
    double getPayout(double bet, bool isBlackjack) const {
        if (isBlackjack) {
            return bet + (bet * blackjackPayout);
        }
        return bet * 2;
    }
};

} // namespace blackjack