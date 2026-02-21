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

    /** Rule presets (common house rules). */
    static GameRules vegasStrip() {
        GameRules r;
        r.numDecks = 6;
        r.dealerHitsSoft17 = false;
        r.blackjackPayout = 1.5;
        r.doubleAfterSplit = true;
        r.surrender = false;
        return r;
    }
    static GameRules downtown() {
        GameRules r;
        r.numDecks = 2;
        r.dealerHitsSoft17 = true;
        r.blackjackPayout = 1.5;
        r.doubleAfterSplit = true;
        r.surrender = true;
        return r;
    }
    static GameRules atlanticCity() {
        GameRules r;
        r.numDecks = 8;
        r.dealerHitsSoft17 = false;
        r.blackjackPayout = 1.5;
        r.doubleAfterSplit = true;
        r.surrender = true;
        return r;
    }
    static GameRules european() {
        GameRules r;
        r.numDecks = 6;
        r.dealerHitsSoft17 = false;
        r.blackjackPayout = 1.5;
        r.surrender = false;
        return r;
    }
    static GameRules singleDeck() {
        GameRules r;
        r.numDecks = 1;
        r.dealerHitsSoft17 = true;
        r.blackjackPayout = 1.5;
        return r;
    }
};

} // namespace blackjack