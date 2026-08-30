#include "StrategyChart.hpp"
#include <iomanip>

#ifdef _WIN32
#include <io.h>
#define ISATTY _isatty
#define FILENO _fileno
#else
#include <unistd.h>
#define ISATTY isatty
#define FILENO fileno
#endif

namespace blackjack {
namespace training {

StrategyChart::StrategyChart(double marginThreshold)
    : marginThreshold_(marginThreshold) {}

bool StrategyChart::isTerminal() { return ISATTY(FILENO(stdout)) != 0; }

char StrategyChart::actionChar(ai::Action action) {
  switch (action) {
  case ai::Action::HIT:
    return 'H';
  case ai::Action::STAND:
    return 'S';
  case ai::Action::DOUBLE:
    return 'D';
  case ai::Action::SPLIT:
    return 'P';
  case ai::Action::SURRENDER:
    return 'R';
  default:
    return '?';
  }
}

double
StrategyChart::computeMargin(ai::Agent &agent, const ai::State &state,
                             const std::vector<ai::Action> &validActions) {
  // Same logic as ConvergenceReport::computeQMargin
  if (validActions.size() < 2)
    return 0.0;
  double top1 = -1e30, top2 = -1e30;
  for (ai::Action a : validActions) {
    double q = agent.getQValue(state, a);
    if (q > top1) {
      top2 = top1;
      top1 = q;
    } else if (q > top2) {
      top2 = q;
    }
  }
  return (top2 < -1e29) ? 0.0 : (top1 - top2);
}

std::vector<ai::Action>
StrategyChart::validActionsForState(const ai::State &state,
                                    bool allowSurrender) {
  // Single source of truth: see legalActionsFor in Evaluator.cpp. This used to
  // be a hand-maintained copy and drifted out of sync with both the audit and
  // the game engine.
  GameRules rules;
  rules.surrender = allowSurrender;
  return legalActionsFor(state, rules);
}

void StrategyChart::print(ai::Agent &agent, const BasicStrategy &basicStrategy,
                          std::ostream &out, bool forceNoColor,
                          bool allowSurrender) const {
  bool useColor = !forceNoColor && isTerminal();

  if (useColor) {
    out << "\n" << BOLD << "=== Strategy Chart ===" << RESET << "\n";
    out << "Legend: " << GREEN << "H" << RESET << "=matches basic strategy  "
        << RED << "H" << RESET << "=diverges  " << YELLOW << "H" << RESET
        << "=uncertain (margin<" << marginThreshold_ << ")\n";
  } else {
    out << "\n=== Strategy Chart ===\n";
    out << "Legend: UPPER=matches basic strategy  lower=diverges\n"
        << "  margin<" << marginThreshold_ << " treated as uncertain\n";
  }
  out << "Actions: H=Hit S=Stand D=Double P=Split R=Surrender  .=never trained\n";

  if (useColor) {
    out << "\n" << BOLD << "--- Hard Totals ---" << RESET << "\n";
  } else {
    out << "\n--- Hard Totals ---\n";
  }
  printGrid(agent, basicStrategy, false, out, forceNoColor, allowSurrender);

  if (useColor) {
    out << "\n" << BOLD << "--- Soft Totals ---" << RESET << "\n";
  } else {
    out << "\n--- Soft Totals ---\n";
  }
  printGrid(agent, basicStrategy, true, out, forceNoColor, allowSurrender);
}

void StrategyChart::printGrid(ai::Agent &agent,
                              const BasicStrategy &basicStrategy,
                              bool softTotals, std::ostream &out,
                              bool forceNoColor, bool allowSurrender) const {
  bool useColor = !forceNoColor && isTerminal();

  // Dealer upcards: 2,3,4,5,6,7,8,9,10,A
  // Display header
  out << std::setw(6) << "";
  const int dealerCards[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 1};
  const char *dealerLabels[] = {"2", "3", "4", "5", "6",
                                "7", "8", "9", "T", "A"};
  for (int i = 0; i < 10; ++i) {
    out << std::setw(3) << dealerLabels[i];
  }
  out << "\n";

  // Row range
  int rowStart = softTotals ? 13 : 4;
  int rowEnd = softTotals ? 21 : 21;

  for (int playerTotal = rowStart; playerTotal <= rowEnd; ++playerTotal) {
    // Skip soft totals > 21 or invalid
    out << std::setw(5) << playerTotal << " ";
    for (int i = 0; i < 10; ++i) {
      int dealerCard = dealerCards[i];
      // A strategy chart describes the opening hand, where doubling is legal.
      // Rendering the canDouble=false slot shows the already-hit policy, in
      // which DOUBLE is untrainable and every double cell reads as a hit.
      ai::State state(playerTotal, dealerCard, softTotals, /*canSplit=*/false,
                      /*canDouble=*/true);

      std::vector<ai::Action> valid = validActionsForState(state, allowSurrender);

      bool trained = true;
      for (ai::Action action : valid) {
        if (agent.getVisitCount(state, action) == 0) {
          trained = false;
          break;
        }
      }
      if (!trained) {
        // Never learned here: any argmax would just report Q-table
        // initialisation order.
        out << "  " << '.';
        continue;
      }

      ai::Action agentAction = agent.chooseAction(state, valid, false);
      bool matches = basicStrategy.isCorrectAction(state, agentAction, allowSurrender);
      double margin = computeMargin(agent, state, valid);
      char ch = actionChar(agentAction);

      if (useColor) {
        const char *color;
        if (matches) {
          color = GREEN;
        } else if (margin < marginThreshold_) {
          color = YELLOW;
        } else {
          color = RED;
        }
        out << "  " << color << ch << RESET;
      } else {
        // Plain text: lowercase = diverges, uppercase = matches
        if (!matches)
          ch = static_cast<char>(ch + 32); // lowercase
        out << "  " << ch;
      }
    }
    out << "\n";
  }
}

} // namespace training
} // namespace blackjack