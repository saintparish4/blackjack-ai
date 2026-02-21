#pragma once

#include "GameRules.hpp"
#include "Deck.hpp"
#include "Hand.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace blackjack {

    enum class Outcome {
        PLAYER_WIN,
        PLAYER_BLACKJACK,
        DEALER_WIN,
        PUSH,
        PLAYER_BUST,
        DEALER_BUST,
        SURRENDER
    };

    std::string outcomeToString(Outcome outcome);

    /** Single-player vs dealer; manages state, rules, and dealer play.
     *  Supports one split per round (no resplit); hands played sequentially. */
    class BlackjackGame {
    public:
        explicit BlackjackGame(const GameRules& rules = GameRules{},
                               std::optional<uint32_t> seed = std::nullopt);
        void startRound();

        /** @return true if action was applied. */
        bool hit();
        void stand();

        /** @return true if action was applied. */
        bool doubleDown();

        /** @return true if split was performed (one split only, no resplit). */
        bool split();

        /** @return true if surrender was applied (only on first two cards; gated by rules.surrender). */
        bool surrender();

        bool isRoundComplete() const { return roundComplete_; }

        /** @throws std::logic_error if round not complete. */
        Outcome getOutcome() const;

        /** When round had multiple hands (e.g. after split), returns one outcome per hand. */
        const std::vector<Outcome>& getOutcomes() const { return outcomes_; }

        /** One bool per hand: true if that hand was doubled. Same size as getOutcomes(). */
        const std::vector<bool>& getWasDoubledByHand() const { return doubledByHand_; }

        /** Current hand (single hand index when multiple hands). */
        const Hand& getPlayerHand() const;

        /** hideHoleCard: true to show only upcard (e.g. during player turn). */
        Hand getDealerHand(bool hideHoleCard = false) const;

        bool canDoubleDown() const;
        /** True if current hand can split and no split has been used this round. */
        bool canSplit() const;
        /** True if surrender is allowed (rules + first two cards, single hand). */
        bool canSurrender() const;
        const GameRules& getRules() const { return rules_; }
        void reset();

    private:
        GameRules rules_;
        std::unique_ptr<Deck> deck_;
        std::vector<Hand> playerHands_;
        size_t currentHandIndex_;
        bool splitUsed_;
        Hand dealerHand_;
        bool roundComplete_;
        std::optional<Outcome> outcome_;
        /** One outcome per player hand when round had split; otherwise empty. */
        std::vector<Outcome> outcomes_;
        /** One bool per hand: true if that hand was doubled down. */
        std::vector<bool> doubledByHand_;

        void playDealerHand();
        Outcome determineOutcome(const Hand& playerHand) const;
        void finishRoundAndResolveOutcomes();
        void checkAndReshuffle();
    };

} // namespace blackjack