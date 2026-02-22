#pragma once

#include "../ai/Agent.hpp"
#include "Evaluator.hpp"
#include <iostream>
#include <vector>

namespace blackjack {
namespace training {

class StrategyChart {
public:
  // marginThreshold: Q-value margin below this → yellow (uncertain)
  explicit StrategyChart(double marginThreshold = 0.05);

  void print(ai::Agent &agent, const BasicStrategy &basicStrategy,
             std::ostream &out = std::cout) const;

private:
  double marginThreshold_;

  // ANSI color codes
  static constexpr const char *RESET = "\033[0m";
  static constexpr const char *GREEN = "\033[32m";
  static constexpr const char *RED = "\033[31m";
  static constexpr const char *YELLOW = "\033[33m";
  static constexpr const char *BOLD = "\033[1m";
  static constexpr const char *DIM = "\033[2m";

  static bool isTerminal();

  // Action → single char: H, S, D, P, R
  static char actionChar(ai::Action action);

  // Compute Q-value margin for coloring
  static double computeMargin(ai::Agent &agent, const ai::State &state,
                              const std::vector<ai::Action> &validActions);

  // Build valid action set (same as ConvergenceReport::validActionsForState)
  static std::vector<ai::Action> validActionsForState(const ai::State &state);

  void printGrid(ai::Agent &agent, const BasicStrategy &basicStrategy,
                 bool softTotals, std::ostream &out) const;
};

} // namespace training
} // namespace blackjack