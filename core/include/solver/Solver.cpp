#include "Solver.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace blackjack {
namespace solver {

using ai::Action;

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

/// Farmer compares stand, then hit, then double, with strict "<", so a tie
/// keeps the earlier play. Matching that order keeps the two strategy tables
/// identical even on a (vanishingly rare) exact tie.
constexpr Action kPreference[] = {Action::STAND, Action::HIT, Action::DOUBLE,
                                  Action::SPLIT, Action::SURRENDER};

double standValue(const DealerDist &d, int playerTotal) {
  double v = d.bust - d.blackjack;
  for (int t = 17; t <= 21; ++t) {
    const double p = d.total[static_cast<size_t>(t - 17)];
    if (playerTotal > t) v += p;
    else if (playerTotal < t) v -= p;
  }
  return v;
}

uint64_t keyOf(int r) { return uint64_t{1} << (5 * (r - 1)); }

} // namespace

// ---- ActionValues ----

ActionValues::ActionValues() { v.fill(NaN); }

bool ActionValues::legal(Action a) const { return !std::isnan((*this)[a]); }

Action ActionValues::best() const {
  Action best = Action::STAND;
  double bestValue = -std::numeric_limits<double>::infinity();
  for (Action a : kPreference) {
    if (legal(a) && (*this)[a] > bestValue) {
      best = a;
      bestValue = (*this)[a];
    }
  }
  return best;
}

// ---- Policies ----

Action OptimalPolicy::choose(const HandInfo &info, const ActionValues *unsplit) const {
  if (unsplit == nullptr) {
    throw std::logic_error("OptimalPolicy needs unsplit values");
  }
  // Restrict the unsplit hand's options to what this hand may do: after a
  // split there is no surrender, no resplit and (without DAS) no double.
  ActionValues allowed = *unsplit;
  if (!info.canDouble) allowed[Action::DOUBLE] = NaN;
  if (!info.canSplit) allowed[Action::SPLIT] = NaN;
  if (!info.canSurrender) allowed[Action::SURRENDER] = NaN;
  return allowed.best();
}

ai::State AgentPolicy::stateFor(const HandInfo &info, int countBucket) {
  ai::State s(info.hand.total(), info.upCard, info.hand.soft(), info.canSplit,
              info.canDouble);
  s.countBucket = countBucket;
  return s;
}

Action AgentPolicy::choose(const HandInfo &info, const ActionValues *) const {
  // Same order as GameStateConverter::getValidActions, so an agent that
  // breaks ties by position breaks them the same way here.
  std::vector<Action> valid{Action::HIT, Action::STAND};
  if (info.canDouble) valid.push_back(Action::DOUBLE);
  if (info.canSplit) valid.push_back(Action::SPLIT);
  if (info.canSurrender) valid.push_back(Action::SURRENDER);
  return agent_.chooseAction(stateFor(info, countBucket_), valid, false);
}

// ---- Solver internals ----

/// Values of one composition against every upcard, in one context.
struct Solver::Node {
  bool ready = false;
  std::array<double, 11> stand{};
  std::array<double, 11> hit{};
  std::array<double, 11> dbl{};    ///< Two-card hands only
  std::array<double, 11> chosen{}; ///< The policy's play, where it can be asked
};

/// A context is a base shoe: the full shoe, or the full shoe less one extra
/// pair card, which is where each half of a split draws from.
struct Solver::Context {
  int pairCard = 0; ///< 0 for the unsplit context
  std::unordered_map<uint64_t, Node> nodes;
};

Solver::Solver(const Rules &rules, const Policy &policy)
    : Solver(rules, Shoe::decks(rules.decks), policy) {}

Solver::Solver(const Rules &rules, const Shoe &shoe, const Policy &policy)
    : rules_(rules), shoe_(shoe), policy_(policy),
      main_(std::make_unique<Context>()) {}

Solver::~Solver() = default;

const std::array<DealerDist, 11> &Solver::dealerFor(uint64_t removedKey,
                                                    const Shoe &s) {
  auto it = dealerCache_.find(removedKey);
  if (it != dealerCache_.end()) return it->second;
  return dealerCache_.emplace(removedKey, dealerOutcomesByUpCard(s, rules_.hitsSoft17))
      .first->second;
}

double Solver::probabilityHoleNatural(const Shoe &afterUpCard, int upCard) const {
  if (upCard == ACE) return afterUpCard.p(TEN);
  if (upCard == TEN) return afterUpCard.p(ACE);
  return 0.0;
}

double Solver::pairOfTensSameRank() const {
  // Given two ten-valued cards, the chance they share a rank, with the shoe's
  // tens spread evenly over T, J, Q, K. For a full shoe this is exactly
  // (4D - 1) / (16D - 1), whatever other tens have been dealt face down.
  const double n = shoe_.n[TEN];
  return n > 1 ? (n - 4.0) / (4.0 * (n - 1.0)) : 0.0;
}

Solver::Context &Solver::splitContext(int pairCard) {
  auto &ctx = split_[static_cast<size_t>(pairCard)];
  if (!ctx) {
    ctx = std::make_unique<Context>();
    ctx->pairCard = pairCard;
  }
  return *ctx;
}

Solver::Node &Solver::node(Context &ctx, Composition &hand) {
  Node &n = ctx.nodes[hand.key()];
  if (n.ready) return n;

  Shoe s = shoe_;
  uint64_t removed = hand.key();
  for (int r = 1; r <= 10; ++r) {
    s.n[r] -= hand.c[r];
    s.total -= hand.c[r];
  }
  if (ctx.pairCard != 0) {
    s.remove(ctx.pairCard);
    removed += keyOf(ctx.pairCard);
  }
  const auto &dealer = dealerFor(removed, s);
  const int total = hand.total();
  const bool splitAces = ctx.pairCard == ACE;

  for (int up = 1; up <= 10; ++up) {
    if (s.n[up] == 0) continue;
    n.stand[up] = standValue(dealer[up], total);
    if (splitAces) continue; // split aces take one card and stand

    s.remove(up);
    const double natural = probabilityHoleNatural(s, up);
    double hit = 0.0;
    double dbl = natural; // a peeked natural takes one bet, not the doubled two
    for (int c = 1; c <= 10; ++c) {
      if (s.n[c] == 0) continue;
      const double p = s.p(c);
      hand.add(c);
      if (hand.bust()) {
        hit -= p;
        dbl -= 2.0 * p;
      } else {
        Node &child = node(ctx, hand);
        hit += p * child.chosen[up];
        dbl += p * 2.0 * child.stand[up];
      }
      hand.remove(c);
    }
    s.add(up);
    n.hit[up] = hit;
    n.dbl[up] = hand.cards == 2 ? dbl : NaN;
  }

  // The policy is asked here only where the engine would ask it with no
  // option beyond stand, hit and (after a split, with DAS) double. Opening
  // hands, with split and surrender on offer, are decided in openingValues.
  const bool splitHand = ctx.pairCard != 0;
  const bool decides = splitHand ? hand.cards >= 2 : hand.cards >= 3;
  if (decides) {
    const bool canDouble = splitHand && rules_.doubleAfterSplit && hand.cards == 2;
    for (int up = 1; up <= 10; ++up) {
      if (s.n[up] == 0) continue;
      if (splitAces) {
        n.chosen[up] = n.stand[up];
        continue;
      }
      ActionValues own;
      own[Action::STAND] = n.stand[up];
      own[Action::HIT] = n.hit[up];
      if (canDouble) own[Action::DOUBLE] = n.dbl[up];

      const ActionValues *unsplit = nullptr;
      ActionValues unsplitValues;
      if (policy_.needsUnsplitValues()) {
        if (splitHand) {
          // CDZ-: a split hand plays as the same composition would unsplit.
          Node &u = node(*main_, hand);
          unsplitValues[Action::STAND] = u.stand[up];
          unsplitValues[Action::HIT] = u.hit[up];
          if (canDouble) unsplitValues[Action::DOUBLE] = u.dbl[up];
          unsplit = &unsplitValues;
        } else {
          unsplit = &own;
        }
      }
      const HandInfo info{hand, up, canDouble, false, false, splitHand};
      n.chosen[up] = own[policy_.choose(info, unsplit)];
    }
  }

  n.ready = true;
  return n;
}

double Solver::splitValue(int pairCard, int upCard) {
  Shoe s = shoe_;
  if (s.n[pairCard] < 2) return NaN;
  s.remove(pairCard);
  s.remove(pairCard);
  if (s.n[upCard] == 0) return NaN;
  s.remove(upCard);
  const double natural = probabilityHoleNatural(s, upCard);

  Context &ctx = splitContext(pairCard);
  Composition hand;
  hand.add(pairCard);
  double one = 0.0;
  for (int c = 1; c <= 10; ++c) {
    if (s.n[c] == 0) continue;
    hand.add(c);
    one += s.p(c) * node(ctx, hand).chosen[upCard];
    hand.remove(c);
  }
  // Each hand already loses one bet to a natural; a peek takes only the
  // initial bet, so give one back.
  return 2.0 * one + natural;
}

ActionValues Solver::handValues(const Composition &hand, int upCard) {
  Composition h = hand;
  Node &n = node(*main_, h);
  ActionValues v;
  v[Action::STAND] = n.stand[upCard];
  v[Action::HIT] = n.hit[upCard];
  if (hand.cards == 2) v[Action::DOUBLE] = n.dbl[upCard];
  return v;
}

ActionValues Solver::openingValues(int card1, int card2, int upCard) {
  Composition h;
  h.add(card1);
  h.add(card2);
  Node &n = node(*main_, h);

  Shoe s = shoe_;
  s.remove(card1);
  s.remove(card2);
  s.remove(upCard);
  const double natural = probabilityHoleNatural(s, upCard);

  ActionValues v;
  if (h.total() == 21) {
    // A natural ends the round before anyone decides anything.
    v[Action::STAND] = rules_.blackjackPayout * (1.0 - natural);
    return v;
  }
  v[Action::STAND] = n.stand[upCard];
  v[Action::HIT] = n.hit[upCard];
  v[Action::DOUBLE] = n.dbl[upCard];
  if (card1 == card2) v[Action::SPLIT] = splitValue(card1, upCard);
  if (rules_.surrender) v[Action::SURRENDER] = -0.5 - natural / 2.0;
  return v;
}

ActionValues Solver::peekedValues(const Composition &hand, int upCard) {
  ActionValues v;
  int first = 0, second = 0;
  if (hand.cards == 2) {
    for (int r = 1; r <= 10; ++r) {
      for (int k = 0; k < hand.c[r]; ++k) (first == 0 ? first : second) = r;
    }
    v = openingValues(first, second, upCard);
  } else {
    v = handValues(hand, upCard);
  }
  Shoe s = shoe_;
  for (int r = 1; r <= 10; ++r) {
    s.n[r] -= hand.c[r];
    s.total -= hand.c[r];
  }
  s.remove(upCard);
  const double p = 1.0 - probabilityHoleNatural(s, upCard);
  const bool natural = hand.cards == 2 && hand.total() == 21;
  for (double &x : v.v) {
    if (std::isnan(x)) continue;
    x = natural ? x / p : (x + 1.0 - p) / p;
  }
  return v;
}

Action Solver::openingAction(int card1, int card2, int upCard, bool canSplit) {
  const ActionValues v = openingValues(card1, card2, upCard);
  if (!v.legal(Action::HIT)) return Action::STAND; // a natural: nothing to decide
  Composition h;
  h.add(card1);
  h.add(card2);
  const bool splittable = card1 == card2 && canSplit && v.legal(Action::SPLIT);
  const HandInfo info{h, upCard, true, splittable, rules_.surrender, false};
  return policy_.choose(info, policy_.needsUnsplitValues() ? &v : nullptr);
}

double Solver::expectedValue(int upCard) {
  Shoe s = shoe_;
  if (s.n[upCard] == 0) return NaN;
  s.remove(upCard);
  const double qSameTen = pairOfTensSameRank();

  double ev = 0.0;
  for (int c1 = 1; c1 <= 10; ++c1) {
    if (s.n[c1] == 0) continue;
    const double p1 = s.p(c1);
    s.remove(c1);
    for (int c2 = 1; c2 <= 10; ++c2) {
      if (s.n[c2] == 0) continue;
      const double p = p1 * s.p(c2);
      const ActionValues v = openingValues(c1, c2, upCard);
      auto value = [&](bool canSplit) {
        return v[openingAction(c1, c2, upCard, canSplit)];
      };
      if (c1 == TEN && c2 == TEN && rules_.splitIdenticalRanksOnly) {
        // K,Q is a hard 20 that cannot split; K,K can.
        ev += p * (qSameTen * value(true) + (1.0 - qSameTen) * value(false));
      } else {
        ev += p * value(true);
      }
    }
    s.add(c1);
  }
  return ev;
}

double Solver::expectedValue() {
  double ev = 0.0;
  for (int up = 1; up <= 10; ++up) {
    if (shoe_.n[up] == 0) continue;
    ev += shoe_.p(up) * expectedValue(up);
  }
  return ev;
}

std::array<DealerDist, 11> Solver::dealerOutcomes() const {
  return dealerOutcomesByUpCard(shoe_, rules_.hitsSoft17);
}

} // namespace solver
} // namespace blackjack