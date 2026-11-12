// solve: exact expected values for a set of rules, by combinatorial analysis.
//
// Prints the same three blocks as Eric Farmer's strategy calculator, in the
// same layout, so the two can be compared with diff rather than by eye:
// the two-card strategy table, the EV against each upcard, and the dealer's
// final-hand probabilities. tools/farmer/compare_solver.sh runs that diff.

#include "solver/InfiniteDeck.hpp"
#include "solver/Solver.hpp"
#include "training/Evaluator.hpp"
#include "util/ArgParser.hpp"

#include <cstdio>
#include <string>

using namespace blackjack;
using namespace blackjack::solver;
using ai::Action;

namespace {

// Farmer's getStrategy(): every option ranked above the better of stand and
// hit, then that one, padded to four characters. Upper case if the option
// favours the player, judged after the dealer has peeked.
std::string strategyCell(Solver &solver, int c1, int c2, int up) {
  Composition h;
  h.add(c1);
  h.add(c2);
  const ActionValues v = solver.peekedValues(h, up);

  char options[5];
  double values[5];
  int n = 0;
  auto add = [&](Action a, char upper, char lower) {
    if (!v.legal(a)) return;
    values[n] = v[a];
    options[n++] = v[a] > 0 ? upper : lower;
  };
  add(Action::STAND, 'S', 's');
  add(Action::HIT, 'H', 'h');
  add(Action::DOUBLE, 'D', 'd');
  add(Action::SPLIT, 'P', 'p'); // any two equal values, tens included, as Farmer
  if (v.legal(Action::SURRENDER)) {
    values[n] = -0.5;
    options[n++] = 'r';
  }
  // The same exchange sort Farmer uses, so ties order identically.
  for (int i = 0; i < n - 1; ++i)
    for (int j = i + 1; j < n; ++j)
      if (values[i] < values[j]) {
        std::swap(values[i], values[j]);
        std::swap(options[i], options[j]);
      }
  int last = 0;
  while (options[last] != 'S' && options[last] != 's' && options[last] != 'H' &&
         options[last] != 'h')
    ++last;
  std::string cell(options, options + last + 1);
  cell.resize(4, ' ');
  return cell;
}

void printRow(Solver &solver, const char *label, int c1, int c2) {
  std::printf("%s |", label);
  for (int up = 2; up <= 10; ++up) std::printf(" %s", strategyCell(solver, c1, c2, up).c_str());
  std::printf(" %s\n", strategyCell(solver, c1, c2, ACE).c_str());
}

void printStrategy(Solver &solver) {
  const char *header = "  hand | 2    3    4    5    6    7    8    9    10   A\n"
                       "-----------------------------------------------------------\n";
  std::printf("  Hard |                Dealer's up card\n%s", header);
  for (int count = 19; count >= 5; --count) {
    int card1 = count > 12 ? 10 : count - 2;
    int card2 = count - card1;
    do {
      char label[32];
      std::snprintf(label, sizeof label, "%3d-%2d", card1, card2);
      printRow(solver, label, card1, card2);
    } while (--card1 > ++card2);
    std::printf("       |\n");
  }
  std::printf("\n  Soft |                Dealer's up card\n%s", header);
  for (int card = 9; card >= 2; --card) {
    char label[32];
    std::snprintf(label, sizeof label, "  A-%2d", card);
    printRow(solver, label, ACE, card);
  }
  std::printf("\n  Pair |                Dealer's up card\n%s", header);
  printRow(solver, "  A- A", ACE, ACE);
  for (int card = 10; card >= 2; --card) {
    char label[32];
    std::snprintf(label, sizeof label, "%3d-%2d", card, card);
    printRow(solver, label, card, card);
  }
  std::printf("-----------------------------------------------------------\n");
}

void printUpCardValues(Solver &solver) {
  std::printf("\n    Up  |\n   card | Overall expected value (%%)\n");
  std::printf("---------------------------------\n");
  for (int up = 2; up <= 10; ++up)
    std::printf("%6d  | %14.9f\n", up, solver.expectedValue(up) * 100);
  std::printf("     A  | %14.9f\n", solver.expectedValue(ACE) * 100);
  std::printf("---------------------------------\n");
  std::printf("  Total | %14.9f\n", solver.expectedValue() * 100);
}

void printDealer(const std::array<DealerDist, 11> &d, const Shoe *shoe) {
  std::printf("\n    Up  |              Probability of outcome of dealer's hand\n");
  std::printf("   card |  Bust   |   17    |   18    |   19    |   20    |   21    |Blackjack\n");
  std::printf("-------------------------------------------------------------------------------\n");
  DealerDist all;
  auto row = [](const char *label, const DealerDist &x) {
    std::printf("%s | %.5f", label, x.bust);
    for (double p : x.total) std::printf(" | %.5f", p);
    std::printf(" | %.5f\n", x.blackjack);
  };
  for (int i = 0; i < 10; ++i) {
    const int up = i < 9 ? i + 2 : ACE;
    const double pUp = shoe ? shoe->p(up) : (up == TEN ? 4.0 / 13 : 1.0 / 13);
    char label[32];
    std::snprintf(label, sizeof label, up == ACE ? "     A " : "%6d ", up);
    row(label, d[static_cast<size_t>(up)]);
    all.bust += pUp * d[static_cast<size_t>(up)].bust;
    all.blackjack += pUp * d[static_cast<size_t>(up)].blackjack;
    for (size_t t = 0; t < 5; ++t) all.total[t] += pUp * d[static_cast<size_t>(up)].total[t];
  }
  std::printf("-------------------------------------------------------------------------------\n");
  row("  Total", all);
}

} // namespace

int main(int argc, char *argv[]) {
  util::ArgParser args("solve", "Exact blackjack expected values by combinatorial analysis");
  args.addFlag("decks", "d", "Number of decks", "6");
  args.addBool("h17", "", "Dealer hits soft 17 (default: stands)");
  args.addBool("das", "", "Double after split (the engine does not allow it)");
  args.addBool("surrender", "", "Late surrender");
  args.addFlag("payout", "", "Blackjack payout", "1.5");
  args.addBool("any-ten-split", "", "Split any two tens, as Farmer's CA does "
                                    "(default: identical ranks only, as the engine)");
  args.addFlag("policy", "p", "optimal (CDZ-) or book (the repo's chart)", "optimal");
  args.addBool("infinite", "", "Infinite-deck optimal EV and dealer table only");
  args.addBool("help", "h", "Show this help message");
  if (!args.parse(argc, argv)) return 0;

  Rules rules;
  rules.decks = args.getInt("decks");
  rules.hitsSoft17 = args.getBool("h17");
  rules.doubleAfterSplit = args.getBool("das");
  rules.surrender = args.getBool("surrender");
  rules.blackjackPayout = args.getDouble("payout");
  rules.splitIdenticalRanksOnly = !args.getBool("any-ten-split");

  if (args.getBool("infinite")) {
    std::printf("Infinite deck, %s\n", rules.hitsSoft17 ? "H17" : "S17");
    std::printf("  Total | %14.9f\n", infiniteDeckExpectedValue(rules) * 100);
    printDealer(infiniteDeckDealer(rules.hitsSoft17), nullptr);
    return 0;
  }

  const std::string policyName = args.getString("policy");
  GameRules game;
  game.numDecks = static_cast<size_t>(rules.decks);
  game.dealerHitsSoft17 = rules.hitsSoft17;
  game.surrender = rules.surrender;
  game.blackjackPayout = rules.blackjackPayout;
  training::BasicStrategyAgent book(game);
  OptimalPolicy optimal;
  AgentPolicy bookPolicy(book);
  const Policy *policy = nullptr;
  if (policyName == "optimal") policy = &optimal;
  else if (policyName == "book") policy = &bookPolicy;
  else {
    std::fprintf(stderr, "Unknown policy: %s\n", policyName.c_str());
    return 1;
  }

  Solver solver(rules, *policy);
  std::printf("%d deck%s, %s, %s, one split, %s, BJ pays %g, %s, policy %s\n\n",
              rules.decks, rules.decks > 1 ? "s" : "", rules.hitsSoft17 ? "H17" : "S17",
              rules.doubleAfterSplit ? "DAS" : "NDAS",
              rules.surrender ? "late surrender" : "no surrender", rules.blackjackPayout,
              rules.splitIdenticalRanksOnly ? "identical-rank splits" : "any ten pair splits",
              policyName.c_str());
  if (policyName == "optimal") printStrategy(solver);
  printUpCardValues(solver);
  const Shoe shoe = solver.shoe();
  printDealer(solver.dealerOutcomes(), &shoe);
  return 0;
}