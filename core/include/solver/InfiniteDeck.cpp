#include "InfiniteDeck.hpp"

#include <algorithm>

namespace blackjack {
namespace solver {

namespace {

double prob(int r) { return r == TEN ? 4.0 / 13.0 : 1.0 / 13.0; }

void dealerDraw(int hardSum, bool hasAce, int cards, double p, bool h17,
                DealerDist &out) {
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
    out.total[static_cast<size_t>(total - 17)] += p;
    return;
  }
  for (int r = 1; r <= 10; ++r) {
    dealerDraw(hardSum + r, hasAce || r == ACE, cards + 1, p * prob(r), h17, out);
  }
}

/// Player values against one upcard, keyed by (hard sum, holds an ace).
struct Table {
  const DealerDist &dealer;
  double natural; ///< P(dealer natural) given this upcard
  std::array<std::array<double, 2>, 32> best{};
  std::array<std::array<bool, 2>, 32> done{};

  static int total(int hardSum, bool ace) {
    return ace && hardSum + 10 <= 21 ? hardSum + 10 : hardSum;
  }

  double stand(int hardSum, bool ace) const {
    const int t = total(hardSum, ace);
    double v = dealer.bust - dealer.blackjack;
    for (int d = 17; d <= 21; ++d) {
      const double p = dealer.total[static_cast<size_t>(d - 17)];
      if (t > d) v += p;
      else if (t < d) v -= p;
    }
    return v;
  }

  double hit(int hardSum, bool ace) {
    double v = 0.0;
    for (int r = 1; r <= 10; ++r) {
      const int h = hardSum + r;
      v += prob(r) * (h > 21 ? -1.0 : bestOf(h, ace || r == ACE));
    }
    return v;
  }

  double dbl(int hardSum, bool ace) const {
    double v = natural;
    for (int r = 1; r <= 10; ++r) {
      const int h = hardSum + r;
      v += prob(r) * (h > 21 ? -2.0 : 2.0 * stand(h, ace || r == ACE));
    }
    return v;
  }

  /// Best of stand and hit: a hand that can no longer double.
  double bestOf(int hardSum, bool ace) {
    auto &slot = best[static_cast<size_t>(hardSum)][ace ? 1 : 0];
    auto &flag = done[static_cast<size_t>(hardSum)][ace ? 1 : 0];
    if (!flag) {
      slot = std::max(stand(hardSum, ace), hit(hardSum, ace));
      flag = true;
    }
    return slot;
  }
};

} // namespace

std::array<DealerDist, 11> infiniteDeckDealer(bool hitsSoft17) {
  std::array<DealerDist, 11> all{};
  for (int up = 1; up <= 10; ++up) {
    dealerDraw(up, up == ACE, 1, 1.0, hitsSoft17, all[static_cast<size_t>(up)]);
  }
  return all;
}

double infiniteDeckExpectedValue(const Rules &rules) {
  const auto dealer = infiniteDeckDealer(rules.hitsSoft17);
  double ev = 0.0;
  for (int up = 1; up <= 10; ++up) {
    const double natural = up == ACE ? prob(TEN) : up == TEN ? prob(ACE) : 0.0;
    Table t{dealer[static_cast<size_t>(up)], natural};

    // One half of a split: the pair card plus one draw, played on.
    auto splitHalf = [&](int x) {
      double v = 0.0;
      for (int r = 1; r <= 10; ++r) {
        const int h = x + r;
        const bool ace = x == ACE || r == ACE;
        double w;
        if (x == ACE) {
          w = t.stand(h, ace);
        } else {
          w = std::max(t.stand(h, ace), t.hit(h, ace));
          if (rules.doubleAfterSplit) w = std::max(w, t.dbl(h, ace));
        }
        v += prob(r) * w;
      }
      return v;
    };

    double evUp = 0.0;
    for (int c1 = 1; c1 <= 10; ++c1) {
      for (int c2 = 1; c2 <= 10; ++c2) {
        const int h = c1 + c2;
        const bool ace = c1 == ACE || c2 == ACE;
        double v;
        if (Table::total(h, ace) == 21) {
          v = rules.blackjackPayout * (1.0 - natural);
        } else {
          v = std::max({t.stand(h, ace), t.hit(h, ace), t.dbl(h, ace)});
          if (c1 == c2) v = std::max(v, 2.0 * splitHalf(c1) + natural);
          if (rules.surrender) v = std::max(v, -0.5 - natural / 2.0);
        }
        evUp += prob(c1) * prob(c2) * v;
      }
    }
    ev += prob(up) * evUp;
  }
  return ev;
}

} // namespace solver
} // namespace blackjack