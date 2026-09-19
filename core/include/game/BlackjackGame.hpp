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
     *  Supports one split per round (no resplit); hands played sequentially.
     *  Split aces receive one card each and a 21 after a split is not a
     *  natural. No double after split. */
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

        /** @return true if split was performed (one split only, no resplit).
         *  Splitting aces deals one card to each and completes the round. */
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

        /** Every player hand this round: one, or two after a split. */
        const std::vector<Hand>& getPlayerHands() const { return playerHands_; }

        /** Index into getPlayerHands() of the hand being played. */
        size_t getCurrentHandIndex() const { return currentHandIndex_; }

        /** hideHoleCard: true to show only upcard (e.g. during player turn). */
        Hand getDealerHand(bool hideHoleCard = false) const;

        /**
         * @brief Reshuffle now if the shoe has passed its penetration.
         *
         * startRound() does this itself. Call it first when a decision has
         * to be made before the deal, such as a bet sized by the count, so
         * the count it reads belongs to the shoe the round is dealt from.
         * @return true if the shoe was reshuffled.
         */
        bool reshuffleIfDue();

        /** Hi-Lo running count of every card the player has seen since the
         *  last shuffle. The dealer's hole card joins it only once revealed. */
        int getRunningCount() const { return runningCount_; }

        /** Decks the player hasn't seen yet: the undealt shoe plus a hole card
         *  still face down. */
        double getDecksRemaining() const;

        /** Running count per unseen deck: the standard Hi-Lo true count. */
        double getTrueCount() const;

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

        int runningCount_ = 0;
        bool holeCounted_ = true;

        /** Deal a card the player sees, adding it to the running count. */
        Card dealVisible();
        /** Count the dealer's hole card once it is turned over. */
        void revealHoleCard();

        void playDealerHand();
        Outcome determineOutcome(const Hand& playerHand) const;
        void finishRoundAndResolveOutcomes();
        void checkAndReshuffle();
    };

} // namespace blackjack