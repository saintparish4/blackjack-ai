#pragma once

#include <cstddef>

namespace blackjack {

/**
 * House rules and table config: only what BlackjackGame actually reads.
 *
 * Some rules are not configurable because the engine plays exactly one
 * version of them, and a field that claims otherwise is worse than no field.
 * This struct used to carry doubleAfterSplit, resplitAces and maxSplits; every
 * preset set them and nothing ever read them. The fixed rules are:
 *
 *  - The dealer peeks: a dealer natural ends the round before the player acts.
 *  - One split per round, no resplits. Split aces take one card each; a 21
 *    after a split is not a natural.
 *  - Only identical ranks split (K,K yes; K,Q no).
 *  - No double after a split. Doubling is allowed on any first two cards.
 *  - Surrender, when offered, is late (after the peek).
 *
 * solver::Rules::fromGame() maps these onto the exact solver, and
 * training/EngineAudit.hpp checks that the engine plays the rules it says
 * it plays.
 */
struct GameRules {
    size_t numDecks = 6;
    bool dealerHitsSoft17 = true;
    double blackjackPayout = 1.5;  // 3:2 = 1.5, 6:5 = 1.2
    bool surrender = false;
    double penetration = 0.75;  // fraction of shoe dealt before reshuffle
    /// Largest bet the player may place, in units of the minimum bet. 1 means
    /// flat betting, which every non-counting run uses.
    int maxBetUnits = 1;

    /** Returns stake + winnings (blackjack uses blackjackPayout multiplier). */
    double getPayout(double bet, bool isBlackjack) const {
        if (isBlackjack) {
            return bet + (bet * blackjackPayout);
        }
        return bet * 2;
    }

    /** Rule presets (common house rules). Each one differs from the others in
     *  at least one rule the engine reads; a preset that duplicates another
     *  only pretends to offer a choice. */
    static GameRules vegasStrip() {
        GameRules r;
        r.numDecks = 6;
        r.dealerHitsSoft17 = false;
        r.blackjackPayout = 1.5;
        r.surrender = false;
        return r;
    }
    static GameRules downtown() {
        GameRules r;
        r.numDecks = 2;
        r.dealerHitsSoft17 = true;
        r.blackjackPayout = 1.5;
        r.surrender = true;
        return r;
    }
    static GameRules atlanticCity() {
        GameRules r;
        r.numDecks = 8;
        r.dealerHitsSoft17 = false;
        r.blackjackPayout = 1.5;
        r.surrender = true;
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