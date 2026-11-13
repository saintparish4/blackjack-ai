// policy_ev: the exact EV of a total-dependent policy, computed by Farmer's CA.
//
//   policy_ev DECKS H17 DAS SURRENDER POLICY_FILE      (y/n flags)
//
// POLICY_FILE is what solver::writeFarmerPolicy() writes: one line per
// state, "total soft up double split surrender action". Each line becomes
// one of Farmer's IndexStrategy entries with no count variation, so his
// BJPlayer plays exactly that policy and reports its EV. This file is ours
// (MIT); it is compiled against Farmer's sources (GPL-3.0), which
// run_farmer.sh fetches, and is never linked into the repository's targets.

#include "blackjack.h"
#include "indices.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: policy_ev DECKS H17 DAS SURRENDER POLICY_FILE\n");
    return 1;
  }
  const int decks = std::stoi(argv[1]);
  auto yes = [](const char *s) { return s[0] == 'y' || s[0] == 'Y'; };
  const bool h17 = yes(argv[2]), das = yes(argv[3]), surrender = yes(argv[4]);

  BJShoe shoe(decks);
  // hitSoft17, doubleAnyTotal, double9, doubleSoft, doubleAfterHit,
  // doubleAfterSplit, resplit, resplitAces, lateSurrender, payoff
  BJRules rules(h17, true, true, true, false, das, false, false, surrender, 1.5);
  // All-zero tags: the true count is always 0, so each entry's single play
  // (valid for every count below +infinity) is the play. Both objects hold
  // large fixed arrays (BJPlayer enumerates every player hand), so they live
  // on the heap, as in Farmer's own strategy.cpp.
  auto strategy = std::make_unique<IndexStrategy>(rules, std::vector<double>(11, 0.0), 1, shoe);

  std::ifstream in(argv[5]);
  if (!in) {
    std::fprintf(stderr, "cannot read %s\n", argv[5]);
    return 1;
  }
  const std::vector<double> always(1, std::numeric_limits<double>::infinity());
  int total, soft, up, dbl, split, surr, action, lines = 0;
  while (in >> total >> soft >> up >> dbl >> split >> surr >> action) {
    strategy->setOption(total, soft != 0, up, dbl != 0, split != 0, surr != 0, always,
                       std::vector<int>(1, action));
    ++lines;
  }

  BJProgress progress;
  auto player = std::make_unique<BJPlayer>(shoe, rules, *strategy, progress);
  std::printf("%d policy lines, EV %.9f%%\n", lines, player->getValue() * 100);
  return 0;
}
