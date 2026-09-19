#include "StrategyChart.hpp"
#include <iomanip>
#include <sstream>

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

  // Formatted separately: the caller's stream may still carry a fixed
  // precision from earlier output, which once printed this 0.05 as "0.1".
  std::ostringstream threshold;
  threshold << marginThreshold_;

  if (useColor) {
    out << "\n" << BOLD << "=== Strategy Chart ===" << RESET << "\n";
    out << "Legend: " << GREEN << "H" << RESET << "=matches basic strategy  "
        << RED << "H" << RESET << "=diverges  " << YELLOW << "H" << RESET
        << "=uncertain (margin<" << threshold.str() << ")\n";
  } else {
    out << "\n=== Strategy Chart ===\n";
    out << "Legend: UPPER=matches basic strategy  lower=diverges\n"
        << "  margin<" << threshold.str()
        << " is shown as uncertain in the colour chart\n";
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

  if (useColor) {
    out << "\n" << BOLD << "--- Pairs ---" << RESET << "\n";
  } else {
    out << "\n--- Pairs ---\n";
  }
  printPairGrid(agent, basicStrategy, out, forceNoColor, allowSurrender);
}

namespace {
const int kDealerCards[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 1};
const char *kDealerLabels[] = {"2", "3", "4", "5", "6", "7", "8", "9", "T", "A"};
} // namespace

void StrategyChart::printHeader(std::ostream &out) {
  out << std::setw(6) << "";
  for (const char *label : kDealerLabels) {
    out << std::setw(3) << label;
  }
  out << "\n";
}

void StrategyChart::printCell(ai::Agent &agent,
                              const BasicStrategy &basicStrategy,
                              const ai::State &cell, std::ostream &out,
                              bool useColor, bool allowSurrender) const {
  // A counting agent's chart is its policy at a true count of zero.
  const ai::State state = auditView(agent, cell);
  std::vector<ai::Action> valid = validActionsForState(state, allowSurrender);

  for (ai::Action action : valid) {
    if (agent.getVisitCount(state, action) == 0) {
      // Never learned here: any argmax would just report Q-table
      // initialisation order.
      out << "  " << '.';
      return;
    }
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

void StrategyChart::printPairGrid(ai::Agent &agent,
                                  const BasicStrategy &basicStrategy,
                                  std::ostream &out, bool forceNoColor,
                                  bool allowSurrender) const {
  bool useColor = !forceNoColor && isTerminal();
  printHeader(out);
  // Same row order as a printed chart: A,A first, then tens down to deuces.
  const int ranks[] = {1, 10, 9, 8, 7, 6, 5, 4, 3, 2};
  for (int rank : ranks) {
    const std::string label =
        rank == 1 ? "A,A" : rank == 10 ? "T,T" : std::to_string(rank) + "," + std::to_string(rank);
    out << std::setw(5) << label << " ";
    for (int dealerCard : kDealerCards) {
      const bool aces = (rank == 1);
      ai::State state(aces ? 12 : 2 * rank, dealerCard, aces, /*canSplit=*/true,
                      /*canDouble=*/true);
      printCell(agent, basicStrategy, state, out, useColor, allowSurrender);
    }
    out << "\n";
  }
}

void StrategyChart::printGrid(ai::Agent &agent,
                              const BasicStrategy &basicStrategy,
                              bool softTotals, std::ostream &out,
                              bool forceNoColor, bool allowSurrender) const {
  bool useColor = !forceNoColor && isTerminal();

  printHeader(out);

  int rowStart = softTotals ? 13 : 4;
  int rowEnd = 21;

  for (int playerTotal = rowStart; playerTotal <= rowEnd; ++playerTotal) {
    out << std::setw(5) << playerTotal << " ";
    for (int dealerCard : kDealerCards) {
      // A strategy chart describes the opening hand, where doubling is legal.
      // Rendering the canDouble=false slot shows the already-hit policy, in
      // which DOUBLE is untrainable and every double cell reads as a hit.
      ai::State state(playerTotal, dealerCard, softTotals, /*canSplit=*/false,
                      /*canDouble=*/true);
      printCell(agent, basicStrategy, state, out, useColor, allowSurrender);
    }
    out << "\n";
  }
}

} // namespace training
} // namespace blackjack