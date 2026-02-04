#pragma once

#include "GameRules.hpp"
#include "Deck.hpp"
#include "Hand.hpp"
#include <memory>
#include <optional>

namespace blackjack {

    enum class Outcome {
        PLAYER_WIN,
        PLAYER_BLACKJACK,
        DEALER_WIN,
        PUSH,
        PLAYER_BUST,
        DEALER_BUST
    };

    std::string outcomeToString(Outcome outcome);

    /** Single-player vs dealer; manages state, rules, and dealer play. */
    class BlackjackGame {
    public:
        explicit BlackjackGame(const GameRules& rules = GameRules{});
        void startRound();

        /** @return true if action was applied. */
        bool hit();
        void stand();

        /** @return true if action was applied. */
        bool doubleDown();

        bool isRoundComplete() const { return roundComplete_; }

        /** @throws std::logic_error if round not complete. */
        Outcome getOutcome() const;

        const Hand& getPlayerHand() const { return playerHand_; }

        /** hideHoleCard: true to show only upcard (e.g. during player turn). */
        Hand getDealerHand(bool hideHoleCard = false) const;

        bool canDoubleDown() const;
        const GameRules& getRules() const { return rules_; }
        void reset();

    private:
        GameRules rules_;
        std::unique_ptr<Deck> deck_;
        Hand playerHand_;
        Hand dealerHand_;
        bool roundComplete_;
        std::optional<Outcome> outcome_;
        int handCount_;

        void playDealerHand();
        Outcome determineOutcome() const;
        void checkAndReshuffle();
    };

} // namespace blackjack