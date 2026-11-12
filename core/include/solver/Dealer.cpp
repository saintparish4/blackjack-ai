#include "Dealer.hpp"

#include <cstddef>

namespace blackjack {
namespace solver {

namespace {

struct DealerWalk {
  Shoe &shoe;
  bool h17;
  DealerDist &out;

  // hardSum counts aces as 1; hasAce lets one of them count as 11.
  void draw(int hardSum, bool hasAce, int cards, double p) {
    const bool soft = hasAce && hardSum + 10 <= 21;
    const int total = soft ? hardSum + 10 : hardSum;

    if (cards == 2 && total == 21) {
      out.blackjack += p;
      return;
    }
    if (total > 21) {
      out.bust += p;
      return;
    }
    if (total > 17 || (total == 17 && !(soft && h17))) {
      out.total[static_cast<std::size_t>(total - 17)] += p;
      return;
    }
    for (int r = 1; r <= 10; ++r) {
      if (shoe.n[r] == 0) continue;
      const double pr = p * shoe.p(r);
      shoe.remove(r);
      draw(hardSum + r, hasAce || r == ACE, cards + 1, pr);
      shoe.add(r);
    }
  }
};

} // namespace

DealerDist dealerOutcomes(Shoe &shoe, int upCard, bool hitsSoft17) {
  DealerDist out;
  DealerWalk walk{shoe, hitsSoft17, out};
  walk.draw(upCard, upCard == ACE, 1, 1.0);
  return out;
}

std::array<DealerDist, 11> dealerOutcomesByUpCard(const Shoe &shoe, bool hitsSoft17) {
  std::array<DealerDist, 11> all{};
  Shoe s = shoe;
  for (int up = 1; up <= 10; ++up) {
    if (s.n[up] == 0) continue;
    s.remove(up);
    all[static_cast<std::size_t>(up)] = dealerOutcomes(s, up, hitsSoft17);
    s.add(up);
  }
  return all;
}

} // namespace solver
} // namespace blackjack