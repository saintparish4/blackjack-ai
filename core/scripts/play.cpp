#include "ai/GameStateConverter.hpp"
#include "ai/QLearningAgent.hpp"
#include "game/BlackjackGame.hpp"
#include "game/GameRules.hpp"
#include "training/ConvergenceReport.hpp"
#include "training/Evaluator.hpp"
#include "training/StrategyChart.hpp"
#include "util/ArgParser.hpp"
#include "util/ConfigParser.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;
using namespace blackjack::util;

// ---------------------------------------------------------------------------
// ANSI color helpers
// ---------------------------------------------------------------------------
namespace color {
constexpr const char *RESET  = "\033[0m";
constexpr const char *RED    = "\033[31m";
constexpr const char *GREEN  = "\033[32m";
constexpr const char *YELLOW = "\033[33m";
constexpr const char *CYAN   = "\033[36m";
constexpr const char *BOLD   = "\033[1m";
constexpr const char *DIM    = "\033[2m";
} // namespace color

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
void playHumanMode(BlackjackGame &game, int numHands, bool beginnerMode);
void playAIMode(BlackjackGame &game, QLearningAgent &agent, int numHands, bool beginnerMode);
void playAdvisorMode(BlackjackGame &game, QLearningAgent &agent, int numHands, bool beginnerMode);
void printReport(QLearningAgent &agent, const GameRules &rules);

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

static GameRules rulesFromPreset(const std::string &preset) {
    if (preset == "vegas-strip")   return GameRules::vegasStrip();
    if (preset == "downtown")      return GameRules::downtown();
    if (preset == "atlantic-city") return GameRules::atlanticCity();
    if (preset == "european")      return GameRules::european();
    if (preset == "single-deck")   return GameRules::singleDeck();
    std::cerr << "Warning: unknown rules preset '" << preset
              << "', falling back to default rules.\n";
    return GameRules{};
}

// ---------------------------------------------------------------------------
// Beginner helpers
// ---------------------------------------------------------------------------

/** Print a friendly introduction to blackjack rules and this program. */
void printWelcome(const std::string &mode) {
    std::cout << color::BOLD << color::CYAN
              << "╔══════════════════════════════════════════════════════╗\n"
              << "║          Welcome to Blackjack AI  (Beginner Mode)   ║\n"
              << "╚══════════════════════════════════════════════════════╝\n"
              << color::RESET << "\n";

    std::cout << color::BOLD << "The Goal:\n" << color::RESET;
    std::cout << "  Get a card total closer to 21 than the dealer — without going over.\n"
              << "  Going over 21 is called a \"bust\" and you lose immediately.\n\n";

    std::cout << color::BOLD << "Card Values:\n" << color::RESET;
    std::cout << "  2 - 10 = the number on the card (a 7 is worth 7, an 8 is worth 8, etc.)\n"
              << "  Jack, Queen, King = 10 each\n"
              << "  Ace   = 11 (or 1 if 11 would bust you — this is called a \"soft\" hand)\n\n";

    std::cout << color::BOLD << "How to Win:\n" << color::RESET;
    std::cout << "  - Beat the dealer's total without busting\n"
              << "  - Dealer busts and you haven't → you win\n"
              << "  - Tie (\"push\") → your bet is returned\n"
              << "  - Blackjack (Ace + 10-value on first two cards) → pays 1.5x!\n\n";

    std::cout << color::BOLD << "Your Actions:\n" << color::RESET;
    std::cout << "  H = Hit       — take another card\n"
              << "  S = Stand     — keep your current total, end your turn\n"
              << "  D = Double    — double your bet and receive exactly one more card\n"
              << "  P = Split     — if you have a pair, split into two separate hands\n"
              << "  R = Surrender — fold the hand and get half your bet back\n\n";

    if (mode == "advisor") {
        std::cout << color::BOLD << "Advisor Mode:\n" << color::RESET;
        std::cout << "  The AI will suggest a move each turn based on what it learned\n"
                  << "  from millions of simulated games. You decide whether to follow it.\n\n";
    } else if (mode == "ai") {
        std::cout << color::BOLD << "AI Mode:\n" << color::RESET;
        std::cout << "  Watch the AI play. It will explain each decision it makes.\n\n";
    } else {
        std::cout << color::BOLD << "Human Mode:\n" << color::RESET;
        std::cout << "  You're playing solo. You start with $100. Each hand bets $10.\n\n";
    }

    std::cout << color::DIM << "─────────────────────────────────────────────────────\n"
              << color::RESET;
}

/** Return a plain-English sentence describing the outcome. */
const char *friendlyOutcome(Outcome o) {
    switch (o) {
    case Outcome::PLAYER_WIN:       return "You win!";
    case Outcome::PLAYER_BLACKJACK: return "Blackjack! You win 1.5x your bet!";
    case Outcome::DEALER_BUST:      return "You win! The dealer went over 21.";
    case Outcome::PUSH:             return "Push — it's a tie. Your bet is returned.";
    case Outcome::DEALER_WIN:       return "Dealer wins this one.";
    case Outcome::PLAYER_BUST:      return "Busted — you went over 21.";
    case Outcome::SURRENDER:        return "Surrendered — half your bet is returned.";
    default:                        return "Round over.";
    }
}

/** How much the hand changes the bankroll given outcome and whether doubled. */
int chipDelta(Outcome o, bool wasDoubled) {
    int bet = wasDoubled ? 20 : 10;
    switch (o) {
    case Outcome::PLAYER_WIN:       return bet;
    case Outcome::PLAYER_BLACKJACK: return 15;   // 1.5x on base $10 always
    case Outcome::DEALER_BUST:      return bet;
    case Outcome::PUSH:             return 0;
    case Outcome::DEALER_WIN:       return -bet;
    case Outcome::PLAYER_BUST:      return -bet;
    case Outcome::SURRENDER:        return -5;
    default:                        return 0;
    }
}

/** Return a confidence label derived from Q-value margin. */
const char *confidenceLabel(double margin) {
    if (margin > 0.15) return "High";
    if (margin > 0.05) return "Medium";
    return "Low";
}

/** Return a plain-English reason for the AI's recommendation. */
std::string beginnerActionReason(Action action, const State &state) {
    int dealer = (state.dealerUpCard == 1) ? 11 : state.dealerUpCard;
    int player = state.playerTotal;

    if (action == Action::SURRENDER) {
        return "Your odds are very poor here — cutting losses is the right call.";
    }
    if (action == Action::DOUBLE) {
        if (player == 11) return "11 is a great doubling spot — most cards give you 21 or close.";
        if (player == 10) return "10 is strong for doubling — you're likely to land near 20.";
        if (player == 9 && dealer >= 3 && dealer <= 6)
            return "9 vs a weak dealer card is a good chance to double and profit.";
        return "The math favors risking more here — doubling is the right move.";
    }
    if (action == Action::SPLIT) {
        return "Splitting turns a weak hand into two chances to win.";
    }
    if (action == Action::STAND) {
        if (player >= 17)
            return "You're in solid territory — any new card risks busting you.";
        if (dealer >= 2 && dealer <= 6)
            return "Dealer's weak card means they'll likely bust on their own — no need to risk.";
        return "Standing gives the dealer a chance to bust — the math favors it here.";
    }
    // HIT
    if (dealer >= 7) {
        return "Dealer's strong card means you need more to compete — take a card.";
    }
    if (player <= 11) {
        return "You can't bust from here — always safe to take another card.";
    }
    return "Taking another card is the best play in this spot.";
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/** Display a hand with card symbols, total, and optional soft-hand explanation. */
void displayHand(const std::string &label, const Hand &hand,
                 bool showTotal = true, bool beginnerMode = false) {
    std::cout << color::BOLD << label << ": " << color::RESET;
    for (const auto &card : hand.getCards()) {
        std::cout << card.toShortString() << " ";
    }
    if (showTotal) {
        auto val = hand.getValue();
        std::cout << color::DIM << "(" << val.total;
        if (val.isSoft) {
            std::cout << " soft";
            if (beginnerMode) {
                std::cout << color::RESET << color::YELLOW
                          << " — Ace counts as 11; flips to 1 if you'd bust"
                          << color::RESET << color::DIM;
            }
        }
        std::cout << ")" << color::RESET;
    }
    std::cout << "\n";
}

/** Prompt the user to choose one of the valid actions; loops until valid. */
Action getUserAction(const std::vector<Action> &validActions, bool beginnerMode) {
    if (beginnerMode) {
        std::cout << color::YELLOW << "\nYour move:" << color::RESET << "\n";
        for (Action a : validActions) {
            switch (a) {
            case Action::HIT:
                std::cout << "  H = Hit        (take another card)\n"; break;
            case Action::STAND:
                std::cout << "  S = Stand      (keep what you have, end your turn)\n"; break;
            case Action::DOUBLE:
                std::cout << "  D = Double     (double your bet, receive one more card)\n"; break;
            case Action::SPLIT:
                std::cout << "  P = Split      (split your pair into two separate hands)\n"; break;
            case Action::SURRENDER:
                std::cout << "  R = Surrender  (give up and get half your bet back)\n"; break;
            }
        }
        std::cout << "Enter choice: ";
    } else {
        std::cout << color::YELLOW << "Action? " << color::RESET << "[";
        for (size_t i = 0; i < validActions.size(); ++i) {
            if (i > 0) std::cout << "/";
            switch (validActions[i]) {
            case Action::HIT:       std::cout << "H"; break;
            case Action::STAND:     std::cout << "S"; break;
            case Action::DOUBLE:    std::cout << "D"; break;
            case Action::SPLIT:     std::cout << "P"; break;
            case Action::SURRENDER: std::cout << "R"; break;
            }
        }
        std::cout << "]: ";
    }

    while (true) {
        std::string input;
        std::getline(std::cin, input);
        if (input.empty()) {
            if (beginnerMode) std::cout << "Enter choice: ";
            else              std::cout << "Invalid. Try again: ";
            continue;
        }
        char c = static_cast<char>(std::toupper(static_cast<unsigned char>(input[0])));

        for (Action a : validActions) {
            char expected = '?';
            switch (a) {
            case Action::HIT:       expected = 'H'; break;
            case Action::STAND:     expected = 'S'; break;
            case Action::DOUBLE:    expected = 'D'; break;
            case Action::SPLIT:     expected = 'P'; break;
            case Action::SURRENDER: expected = 'R'; break;
            }
            if (c == expected) return a;
        }
        if (beginnerMode) {
            std::cout << "  That's not one of the options above. Try again: ";
        } else {
            std::cout << "Invalid. Try again: ";
        }
    }
}

/** Display Q-values for the current state across valid actions (expert mode). */
void displayQValues(QLearningAgent &agent, const State &state,
                    const std::vector<Action> &validActions) {
    auto qvals = agent.getAllQValues(state);
    std::cout << color::CYAN << "  Q-values: " << color::RESET;
    for (Action a : validActions) {
        double q = qvals[static_cast<size_t>(a)];
        std::cout << actionToString(a) << "=" << std::fixed
                  << std::setprecision(3) << q << "  ";
    }
    std::cout << "\n";
}

/** Compute Q-value margin (top - second) for confidence label. */
double computeQMargin(QLearningAgent &agent, const State &state,
                      const std::vector<Action> &validActions) {
    if (validActions.size() < 2) return 1.0;
    auto qvals = agent.getAllQValues(state);
    double top1 = -1e30, top2 = -1e30;
    for (Action a : validActions) {
        double q = qvals[static_cast<size_t>(a)];
        if (q > top1) { top2 = top1; top1 = q; }
        else if (q > top2) { top2 = q; }
    }
    return (top2 < -1e29) ? 0.0 : (top1 - top2);
}

// ---------------------------------------------------------------------------
// Human mode
// ---------------------------------------------------------------------------
void playHumanMode(BlackjackGame &game, int numHands, bool beginnerMode) {
    if (beginnerMode) {
        printWelcome("human");
    } else {
        std::cout << color::BOLD << "=== Human Play Mode ===" << color::RESET << "\n";
        std::cout << "Keys: H=Hit, S=Stand, D=Double, P=Split, R=Surrender\n\n";
    }

    int wins = 0, losses = 0, pushes = 0;
    int balance = 100;  // beginner chip tracking

    for (int hand = 0; hand < numHands || numHands <= 0; ++hand) {
        std::cout << color::BOLD << "--- Hand " << (hand + 1);
        if (beginnerMode) {
            std::cout << "  |  Balance: $" << balance;
        }
        std::cout << " ---" << color::RESET << "\n";

        if (beginnerMode) {
            std::cout << color::DIM << "  (Betting $10 this hand)\n" << color::RESET;
        }

        game.startRound();

        // Immediate blackjack / natural resolution
        if (game.isRoundComplete()) {
            displayHand("Your hand", game.getPlayerHand(), true, beginnerMode);
            displayHand("Dealer", game.getDealerHand(false), true, beginnerMode);
            Outcome o = game.getOutcome();

            bool isWin  = (o == Outcome::PLAYER_WIN || o == Outcome::PLAYER_BLACKJACK || o == Outcome::DEALER_BUST);
            bool isPush = (o == Outcome::PUSH);
            const char *c = isWin ? color::GREEN : (isPush ? color::YELLOW : color::RED);

            if (beginnerMode) {
                int delta = chipDelta(o, false);
                balance += delta;
                std::cout << c << friendlyOutcome(o) << color::RESET;
                if (delta != 0) {
                    std::cout << color::DIM << "  ($" << (delta > 0 ? "+" : "")
                              << delta << " → Balance: $" << balance << ")" << color::RESET;
                }
                std::cout << "\n\n";
            } else {
                std::cout << c << "Result: " << outcomeToString(o) << color::RESET << "\n\n";
            }

            if (isWin) wins++; else if (isPush) pushes++; else losses++;
            continue;
        }

        displayHand("Dealer shows", game.getDealerHand(true), false, beginnerMode);
        if (beginnerMode) {
            int dealerCard = game.getDealerHand(true).getCards().front().getValue();
            if (dealerCard >= 7 || dealerCard == 1) {
                std::cout << color::DIM
                          << "  Dealer's card is strong — tread carefully.\n"
                          << color::RESET;
            } else {
                std::cout << color::DIM
                          << "  Dealer's card is weak — they may bust on their own.\n"
                          << color::RESET;
            }
        }
        displayHand("Your hand", game.getPlayerHand(), true, beginnerMode);

        while (!game.isRoundComplete()) {
            auto validActions = GameStateConverter::getValidActions(
                game.getPlayerHand(), game.canSplit(),
                game.canDoubleDown(), game.canSurrender());

            Action action = getUserAction(validActions, beginnerMode);
            GameStateConverter::executeAction(action, game);

            if (!game.isRoundComplete()) {
                displayHand("Your hand", game.getPlayerHand(), true, beginnerMode);
            }
        }

        std::cout << "\n";
        displayHand("Dealer hand", game.getDealerHand(false), true, beginnerMode);

        const auto &outcomes   = game.getOutcomes();
        const auto &wasDoubled = game.getWasDoubledByHand();
        for (size_t i = 0; i < outcomes.size(); ++i) {
            if (outcomes.size() > 1) std::cout << "Hand " << (i + 1) << ": ";
            Outcome o    = outcomes[i];
            bool doubled = (i < wasDoubled.size()) && wasDoubled[i];
            bool isWin   = (o == Outcome::PLAYER_WIN || o == Outcome::PLAYER_BLACKJACK || o == Outcome::DEALER_BUST);
            bool isPush  = (o == Outcome::PUSH);
            const char *c = isWin ? color::GREEN : (isPush ? color::YELLOW : color::RED);

            if (beginnerMode) {
                int delta = chipDelta(o, doubled);
                balance += delta;
                std::cout << c << friendlyOutcome(o) << color::RESET;
                if (doubled) std::cout << color::DIM << " (doubled)" << color::RESET;
                if (delta != 0) {
                    std::cout << color::DIM << "  ($" << (delta > 0 ? "+" : "")
                              << delta << " → Balance: $" << balance << ")" << color::RESET;
                }
                std::cout << "\n";
            } else {
                std::cout << c << outcomeToString(o) << color::RESET << "\n";
            }

            if (isWin) wins++; else if (isPush) pushes++; else losses++;
        }
        std::cout << "\n";

        if (numHands <= 0) {
            std::cout << "Continue? [Y/n]: ";
            std::string input;
            std::getline(std::cin, input);
            if (!input.empty() && std::toupper(static_cast<unsigned char>(input[0])) == 'N') break;
        }
    }

    int total = wins + losses + pushes;
    std::cout << "\n" << color::BOLD << "=== Session Summary ===" << color::RESET << "\n";
    std::cout << "Hands: " << total << " | Wins: " << wins
              << " | Losses: " << losses << " | Pushes: " << pushes << "\n";
    if (total > 0) {
        std::cout << "Win rate: " << std::fixed << std::setprecision(1)
                  << (wins * 100.0 / total) << "%\n";
    }
    if (beginnerMode) {
        int profit = balance - 100;
        const char *c = (profit >= 0) ? color::GREEN : color::RED;
        std::cout << "Final balance: $" << balance
                  << "  (" << c << (profit >= 0 ? "+" : "") << profit << color::RESET << ")\n";
    }
}

// ---------------------------------------------------------------------------
// AI mode
// ---------------------------------------------------------------------------
void playAIMode(BlackjackGame &game, QLearningAgent &agent, int numHands, bool beginnerMode) {
    if (beginnerMode) {
        printWelcome("ai");
    } else {
        std::cout << color::BOLD << "=== AI Play Mode ===" << color::RESET << "\n\n";
    }

    int wins = 0, losses = 0, pushes = 0;
    double totalReward = 0.0;

    for (int hand = 0; hand < numHands; ++hand) {
        std::cout << color::BOLD << "--- Hand " << (hand + 1) << " ---"
                  << color::RESET << "\n";

        game.startRound();

        if (game.isRoundComplete()) {
            displayHand("Player", game.getPlayerHand(), true, beginnerMode);
            displayHand("Dealer", game.getDealerHand(false), true, beginnerMode);
            Outcome o = game.getOutcome();
            bool isWin  = (o == Outcome::PLAYER_WIN || o == Outcome::PLAYER_BLACKJACK || o == Outcome::DEALER_BUST);
            bool isPush = (o == Outcome::PUSH);
            const char *c = isWin ? color::GREEN : (isPush ? color::YELLOW : color::RED);
            if (beginnerMode) {
                std::cout << c << friendlyOutcome(o) << color::RESET << "\n\n";
            } else {
                std::cout << "Result: " << outcomeToString(o) << "\n\n";
            }
            if (isWin) wins++; else if (isPush) pushes++; else losses++;
            continue;
        }

        displayHand("Dealer shows", game.getDealerHand(true), false, beginnerMode);

        while (!game.isRoundComplete()) {
            const Hand &playerHand = game.getPlayerHand();
            const Hand &dealerHand = game.getDealerHand(true);

            displayHand("Player", playerHand, true, beginnerMode);

            State state = GameStateConverter::toAIState(
                playerHand, dealerHand, game.canSplit(), game.canDoubleDown());
            auto validActions = GameStateConverter::getValidActions(
                playerHand, game.canSplit(), game.canDoubleDown(), game.canSurrender());

            Action action = agent.chooseAction(state, validActions, false);

            if (beginnerMode) {
                double margin = computeQMargin(agent, state, validActions);
                std::cout << color::GREEN << "  AI plays: " << actionToString(action)
                          << color::RESET
                          << color::DIM << "  (Confidence: " << confidenceLabel(margin) << ")\n"
                          << color::RESET;
                std::cout << color::DIM << "  Why: " << beginnerActionReason(action, state)
                          << "\n" << color::RESET;
            } else {
                displayQValues(agent, state, validActions);
                std::cout << color::GREEN << "  -> " << actionToString(action)
                          << color::RESET << "\n";
            }

            GameStateConverter::executeAction(action, game);
        }

        displayHand("Dealer hand", game.getDealerHand(false), true, beginnerMode);

        const auto &outcomes   = game.getOutcomes();
        const auto &wasDoubled = game.getWasDoubledByHand();
        for (size_t i = 0; i < outcomes.size(); ++i) {
            Outcome o       = outcomes[i];
            bool    doubled = (i < wasDoubled.size()) && wasDoubled[i];
            double  reward  = GameStateConverter::outcomeToReward(o, doubled);
            totalReward += reward;

            bool isWin  = (o == Outcome::PLAYER_WIN || o == Outcome::PLAYER_BLACKJACK || o == Outcome::DEALER_BUST);
            bool isPush = (o == Outcome::PUSH);
            const char *c = isWin ? color::GREEN : (isPush ? color::YELLOW : color::RED);

            if (beginnerMode) {
                std::cout << c << friendlyOutcome(o) << color::RESET;
                if (doubled) std::cout << color::DIM << " (doubled)" << color::RESET;
                std::cout << "\n";
            } else {
                std::cout << c << outcomeToString(o) << color::RESET;
                if (doubled) std::cout << " (doubled)";
                std::cout << " [reward: " << std::showpos << reward << std::noshowpos << "]\n";
            }
            if (isWin) wins++; else if (isPush) pushes++; else losses++;
        }
        std::cout << "\n";
    }

    int total = wins + losses + pushes;
    std::cout << color::BOLD << "=== AI Session Summary ===" << color::RESET << "\n";
    std::cout << "Hands: " << total << " | Wins: " << wins
              << " | Losses: " << losses << " | Pushes: " << pushes << "\n";
    std::cout << "Win rate: " << std::fixed << std::setprecision(1)
              << (wins * 100.0 / std::max(1, total)) << "%\n";
    if (!beginnerMode) {
        std::cout << "Total reward: " << std::fixed << std::setprecision(2)
                  << totalReward << " | Avg: "
                  << (totalReward / std::max(1, numHands)) << "\n";
    }
}

// ---------------------------------------------------------------------------
// Advisor mode
// ---------------------------------------------------------------------------
void playAdvisorMode(BlackjackGame &game, QLearningAgent &agent, int numHands, bool beginnerMode) {
    if (beginnerMode) {
        printWelcome("advisor");
    } else {
        std::cout << color::BOLD << "=== Advisor Mode ===" << color::RESET << "\n";
        std::cout << "You play, AI recommends. Keys: H/S/D/P/R\n\n";
    }

    int agreed = 0, disagreed = 0;
    int wins = 0, losses = 0, pushes = 0;
    int balance = 100;  // beginner chip tracking

    for (int hand = 0; hand < numHands || numHands <= 0; ++hand) {
        std::cout << color::BOLD << "--- Hand " << (hand + 1);
        if (beginnerMode) {
            std::cout << "  |  Balance: $" << balance;
        }
        std::cout << " ---" << color::RESET << "\n";

        if (beginnerMode) {
            std::cout << color::DIM << "  (Betting $10 this hand)\n" << color::RESET;
        }

        game.startRound();

        if (game.isRoundComplete()) {
            displayHand("Player", game.getPlayerHand(), true, beginnerMode);
            displayHand("Dealer", game.getDealerHand(false), true, beginnerMode);
            Outcome o = game.getOutcome();
            bool isWin  = (o == Outcome::PLAYER_WIN || o == Outcome::PLAYER_BLACKJACK || o == Outcome::DEALER_BUST);
            bool isPush = (o == Outcome::PUSH);
            const char *c = isWin ? color::GREEN : (isPush ? color::YELLOW : color::RED);
            if (beginnerMode) {
                int delta = chipDelta(o, false);
                balance += delta;
                std::cout << c << friendlyOutcome(o) << color::RESET;
                if (delta != 0) {
                    std::cout << color::DIM << "  ($" << (delta > 0 ? "+" : "")
                              << delta << " → Balance: $" << balance << ")" << color::RESET;
                }
                std::cout << "\n\n";
            } else {
                std::cout << "Result: " << outcomeToString(o) << "\n\n";
            }
            if (isWin) wins++; else if (isPush) pushes++; else losses++;
            continue;
        }

        displayHand("Dealer shows", game.getDealerHand(true), false, beginnerMode);
        if (beginnerMode) {
            int dealerCard = game.getDealerHand(true).getCards().front().getValue();
            if (dealerCard >= 7 || dealerCard == 1) {
                std::cout << color::DIM
                          << "  Dealer's card is strong — tread carefully.\n"
                          << color::RESET;
            } else {
                std::cout << color::DIM
                          << "  Dealer's card is weak — they may bust on their own.\n"
                          << color::RESET;
            }
        }

        while (!game.isRoundComplete()) {
            displayHand("Your hand", game.getPlayerHand(), true, beginnerMode);

            State state = GameStateConverter::toAIState(
                game.getPlayerHand(), game.getDealerHand(true),
                game.canSplit(), game.canDoubleDown());
            auto validActions = GameStateConverter::getValidActions(
                game.getPlayerHand(), game.canSplit(),
                game.canDoubleDown(), game.canSurrender());

            Action aiAction = agent.chooseAction(state, validActions, false);

            if (beginnerMode) {
                double margin = computeQMargin(agent, state, validActions);
                std::cout << color::CYAN << "  AI recommends: " << actionToString(aiAction)
                          << color::RESET
                          << color::DIM << "  (Confidence: " << confidenceLabel(margin) << ")\n"
                          << color::RESET;
                std::cout << color::DIM << "  Why: " << beginnerActionReason(aiAction, state)
                          << "\n" << color::RESET;
            } else {
                displayQValues(agent, state, validActions);
                std::cout << color::CYAN << "  AI recommends: "
                          << actionToString(aiAction) << color::RESET << "\n";
            }

            Action userAction = getUserAction(validActions, beginnerMode);

            if (userAction == aiAction) {
                std::cout << color::GREEN << "  Good call — you agreed with the AI!"
                          << color::RESET << "\n";
                agreed++;
            } else {
                if (beginnerMode) {
                    std::cout << color::YELLOW << "  You overrode the AI (it wanted "
                              << actionToString(aiAction)
                              << "). Let's see how it plays out!"
                              << color::RESET << "\n";
                } else {
                    std::cout << color::RED << "  (Override — AI wanted "
                              << actionToString(aiAction) << ")" << color::RESET << "\n";
                }
                disagreed++;
            }

            GameStateConverter::executeAction(userAction, game);
        }

        displayHand("Dealer hand", game.getDealerHand(false), true, beginnerMode);

        const auto &outcomes   = game.getOutcomes();
        const auto &wasDoubled = game.getWasDoubledByHand();
        for (size_t i = 0; i < outcomes.size(); ++i) {
            if (outcomes.size() > 1) std::cout << "Hand " << (i + 1) << ": ";
            Outcome o    = outcomes[i];
            bool doubled = (i < wasDoubled.size()) && wasDoubled[i];
            bool isWin   = (o == Outcome::PLAYER_WIN || o == Outcome::PLAYER_BLACKJACK || o == Outcome::DEALER_BUST);
            bool isPush  = (o == Outcome::PUSH);
            const char *c = isWin ? color::GREEN : (isPush ? color::YELLOW : color::RED);

            if (beginnerMode) {
                int delta = chipDelta(o, doubled);
                balance += delta;
                std::cout << c << friendlyOutcome(o) << color::RESET;
                if (doubled) std::cout << color::DIM << " (doubled)" << color::RESET;
                if (delta != 0) {
                    std::cout << color::DIM << "  ($" << (delta > 0 ? "+" : "")
                              << delta << " → Balance: $" << balance << ")" << color::RESET;
                }
                std::cout << "\n";
            } else {
                std::cout << c << outcomeToString(o) << color::RESET << "\n";
            }

            if (isWin) wins++; else if (isPush) pushes++; else losses++;
        }
        std::cout << "\n";

        if (numHands <= 0) {
            std::cout << "Continue? [Y/n]: ";
            std::string input;
            std::getline(std::cin, input);
            if (!input.empty() && std::toupper(static_cast<unsigned char>(input[0])) == 'N') break;
        }
    }

    int total = agreed + disagreed;
    std::cout << "\n" << color::BOLD << "=== Advisor Summary ===" << color::RESET << "\n";
    if (beginnerMode) {
        std::cout << "You followed the AI: " << agreed << " time(s)\n"
                  << "You overrode the AI: " << disagreed << " time(s)\n";
        if (total > 0) {
            std::cout << "Agreement rate: " << std::fixed << std::setprecision(1)
                      << (agreed * 100.0 / total) << "%\n";
            if (agreed * 100.0 / total >= 75.0) {
                std::cout << color::GREEN
                          << "Great job following the AI — that's how you build good instincts!\n"
                          << color::RESET;
            } else if (disagreed > agreed) {
                std::cout << color::YELLOW
                          << "You went your own way a lot. Try following the AI more — "
                             "it's learned from millions of hands!\n"
                          << color::RESET;
            }
        }
        int winsTotal = wins + losses + pushes;
        if (winsTotal > 0) {
            std::cout << "Hands: " << winsTotal << " | Wins: " << wins
                      << " | Losses: " << losses << " | Pushes: " << pushes << "\n";
            std::cout << "Win rate: " << std::fixed << std::setprecision(1)
                      << (wins * 100.0 / winsTotal) << "%\n";
        }
        int profit = balance - 100;
        const char *c = (profit >= 0) ? color::GREEN : color::RED;
        std::cout << "Final balance: $" << balance
                  << "  (" << c << (profit >= 0 ? "+" : "") << profit << color::RESET << ")\n";
    } else {
        std::cout << "Agreed with AI: " << agreed
                  << " | Overrode AI: " << disagreed << "\n";
        if (total > 0) {
            std::cout << "Agreement rate: " << std::fixed << std::setprecision(1)
                      << (agreed * 100.0 / total) << "%\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Report mode
// ---------------------------------------------------------------------------
void printReport(QLearningAgent &agent, const GameRules &rules) {
    Evaluator evaluator(rules);

    StrategyChart chart;
    chart.print(agent, evaluator.getBasicStrategy());

    ConvergenceReport report;
    auto result = report.analyze(agent, evaluator.getBasicStrategy());
    report.print(result);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    ArgParser args("play", "Blackjack Interactive Play");
    args.addFlag("mode",   "m", "Play mode: human, ai, advisor [required unless --report]", "", false);
    args.addFlag("model",  "",  "Path to trained model file");
    args.addFlag("hands",  "n", "Number of hands to play", "10");
    args.addFlag("rules",  "r", "Rule preset name", "vegas-strip");
    args.addFlag("config", "",  "Load INI config file");
    args.addBool("report",   "", "Print strategy chart and convergence report, then exit");
    args.addBool("beginner", "b", "Beginner mode: plain-English explanations, chip balance, AI reasoning");
    args.addBool("help",   "h", "Show this help message");
    if (!args.parse(argc, argv)) return 0;

    if (args.has("config")) {
        ConfigParser cfg;
        try {
            cfg.load(args.getString("config"));
        } catch (const std::exception &e) {
            std::cerr << "Error loading config: " << e.what() << "\n";
            return 1;
        }
    }

    std::string preset = args.getString("rules");
    GameRules rules = rulesFromPreset(preset);

    if (args.has("report")) {
        if (!args.has("model")) {
            std::cerr << "Error: --report requires --model PATH\n";
            return 1;
        }
        QLearningAgent agent;
        agent.load(args.getString("model"));
        printReport(agent, rules);
        return 0;
    }

    if (!args.has("mode")) {
        std::cerr << "Missing required option: --mode\n";
        return 1;
    }
    std::string mode        = args.getString("mode");
    int         numHands    = args.getInt("hands");
    bool        beginnerMode = args.getBool("beginner");

    if (mode != "human" && mode != "ai" && mode != "advisor") {
        std::cerr << "Error: --mode must be human, ai, or advisor\n";
        return 1;
    }

    std::shared_ptr<QLearningAgent> agent;
    if (mode == "ai" || mode == "advisor") {
        if (!args.has("model")) {
            std::cerr << "Error: --model PATH required for " << mode << " mode\n";
            return 1;
        }
        agent = std::make_shared<QLearningAgent>();
        agent->load(args.getString("model"));
        if (!beginnerMode) {
            std::cout << "Loaded model: " << args.getString("model") << "\n";
            std::cout << "States learned: " << agent->getStateCount() << "\n\n";
        }
    }

    BlackjackGame game(rules);

    if (mode == "human")        playHumanMode(game, numHands, beginnerMode);
    else if (mode == "ai")      playAIMode(game, *agent, numHands, beginnerMode);
    else if (mode == "advisor") playAdvisorMode(game, *agent, numHands, beginnerMode);

    return 0;
}
