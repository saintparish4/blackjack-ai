#include "EngineAudit.hpp"

#include "../ai/GameStateConverter.hpp"
#include "../game/BlackjackGame.hpp"
#include "../solver/Solver.hpp"
#include "../util/Seed.hpp"
#include "Evaluator.hpp"

#include <array>
#include <cmath>
#include <string>

namespace blackjack {
namespace training {

using ai::Action;
using solver::ACE;
using solver::TEN;

namespace {

/// Seed streams for the three passes, so each draws its own shoes.
constexpr uint32_t kDealerPass = 101;
constexpr uint32_t kBookPass = 102;
constexpr uint32_t kCutCardPass = 103;

const char *upName(int up) {
  static const char *names[] = {"", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T"};
  return names[up];
}

int upCardOf(const BlackjackGame &game) {
  // getDealerHand() returns a copy: keep the card, not a reference into it.
  const Card c = game.getDealerHand(true).getCards()[0];
  return c.isAce() ? ACE : c.getValue();
}

/** Running mean and variance of one tallied quantity. */
struct Tally {
  uint64_t n = 0;
  double sum = 0.0;
  double sumSq = 0.0;
  void add(double x) {
    ++n;
    sum += x;
    sumSq += x * x;
  }
  double mean() const { return n ? sum / static_cast<double>(n) : 0.0; }
  double stdErr() const {
    if (n < 2) return 0.0;
    const double dn = static_cast<double>(n);
    const double var = (sumSq - dn * mean() * mean()) / (dn - 1.0);
    return var > 0.0 ? std::sqrt(var / dn) : 0.0;
  }
};

AuditLine line(const std::string &name, double exact, const Tally &t) {
  return AuditLine{name, exact, t.mean(), t.stdErr(), t.n};
}

/// The opening categories the book pass splits EV by. Each round lands in
/// exactly one: a natural on either side, or the book's first play.
enum Category { PLAYER_NATURAL, DEALER_NATURAL, STAND, HIT, DOUBLE, SPLIT, SURRENDER, N_CATEGORIES };
const char *kCategoryName[] = {"player natural", "dealer natural", "opening stand",
                               "opening hit", "opening double", "opening split",
                               "opening surrender"};

Category categoryOf(Action a) {
  switch (a) {
  case Action::STAND: return STAND;
  case Action::HIT: return HIT;
  case Action::DOUBLE: return DOUBLE;
  case Action::SPLIT: return SPLIT;
  case Action::SURRENDER: return SURRENDER;
  }
  return STAND;
}

double roundReward(const BlackjackGame &game, double payout) {
  const auto &outcomes = game.getOutcomes();
  const auto &doubled = game.getWasDoubledByHand();
  double r = 0.0;
  for (size_t j = 0; j < outcomes.size(); ++j) {
    r += ai::GameStateConverter::outcomeToReward(outcomes[j],
                                                 j < doubled.size() && doubled[j], payout);
  }
  return r;
}

/** Plays one round with the agent; returns the round's category. */
Category playRound(ai::Agent &agent, BlackjackGame &game) {
  game.startRound();
  if (game.isRoundComplete()) {
    return game.getPlayerHands()[0].isBlackjack() ? PLAYER_NATURAL : DEALER_NATURAL;
  }
  Category category = N_CATEGORIES;
  while (!game.isRoundComplete()) {
    const Hand &hand = game.getPlayerHand();
    const ai::State state = ai::GameStateConverter::toAIState(
        hand, game.getDealerHand(true), game.canSplit(), game.canDoubleDown());
    const auto valid = ai::GameStateConverter::getValidActions(
        hand, game.canSplit(), game.canDoubleDown(), game.canSurrender());
    const Action action = agent.chooseAction(state, valid, false);
    if (category == N_CATEGORIES) category = categoryOf(action);
    ai::GameStateConverter::executeAction(action, game);
  }
  return category;
}

} // namespace

double playOneRound(ai::Agent &agent, BlackjackGame &game) {
  playRound(agent, game);
  return roundReward(game, game.getRules().blackjackPayout);
}

std::vector<AuditLine> auditEngine(const GameRules &rules, const AuditOptions &options) {
  GameRules fresh = rules;
  fresh.penetration = 0.0; // reshuffle before every round: off the top, like the solver

  const solver::Rules exactRules = solver::Rules::fromGame(rules);
  const solver::Shoe full = solver::Shoe::decks(exactRules.decks);
  std::vector<AuditLine> lines;

  // ---- Exact values for the dealer pass ----
  const auto dealerFull = solver::dealerOutcomesByUpCard(full, exactRules.hitsSoft17);
  const double n = full.total;
  const double pPlayerNatural = 2.0 * full.n[ACE] * full.n[TEN] / (n * (n - 1.0));

  // Dealer bust by upcard, over rounds where the player has no natural (the
  // dealer doesn't draw against one). Deal order is player, player, upcard.
  std::array<double, 11> bustWeight{}, bustMass{};
  {
    solver::Shoe s = full;
    for (int c1 = 1; c1 <= 10; ++c1) {
      const double p1 = s.p(c1);
      s.remove(c1);
      for (int c2 = 1; c2 <= 10; ++c2) {
        if (s.n[c2] == 0) continue;
        const double p2 = p1 * s.p(c2);
        s.remove(c2);
        const bool natural = (c1 == ACE && c2 == TEN) || (c1 == TEN && c2 == ACE);
        if (!natural) {
          const auto d = solver::dealerOutcomesByUpCard(s, exactRules.hitsSoft17);
          for (int up = 1; up <= 10; ++up) {
            const double w = p2 * s.p(up);
            bustMass[up] += w;
            bustWeight[up] += w * d[up].bust;
          }
        }
        s.add(c2);
      }
      s.add(c1);
    }
  }

  // ---- Dealer pass: stand on everything ----
  {
    BlackjackGame game(fresh, util::deriveSeed(options.seed, kDealerPass));
    Tally natural;
    std::array<Tally, 11> dealerNatural, bust;
    for (uint64_t i = 0; i < options.rounds; ++i) {
      game.startRound();
      const int up = upCardOf(game);
      const bool playerNatural = game.getPlayerHands()[0].isBlackjack();
      if (!game.isRoundComplete()) game.stand();
      const Hand dealer = game.getDealerHand();
      natural.add(playerNatural ? 1.0 : 0.0);
      dealerNatural[up].add(dealer.isBlackjack() ? 1.0 : 0.0);
      if (!playerNatural) bust[up].add(dealer.isBust() ? 1.0 : 0.0);
    }
    lines.push_back(line("P(player natural)", pPlayerNatural, natural));
    lines.push_back(line("P(dealer natural | up A)", dealerFull[ACE].blackjack, dealerNatural[ACE]));
    lines.push_back(line("P(dealer natural | up T)", dealerFull[TEN].blackjack, dealerNatural[TEN]));
    for (int k = 0; k < 10; ++k) {
      const int up = k < 9 ? k + 2 : ACE;
      lines.push_back(line(std::string("P(dealer busts | up ") + upName(up) + ")",
                           bustWeight[up] / bustMass[up], bust[up]));
    }
  }

  // ---- Exact values for the book pass ----
  BasicStrategyAgent book(rules);
  solver::AgentPolicy bookPolicy(book);
  solver::Solver exact(exactRules, bookPolicy);
  const double bookEv = exact.expectedValue();

  std::array<double, N_CATEGORIES> mass{}, value{};
  {
    const double qSameTen = exact.pairOfTensSameRank();
    solver::Shoe s = full;
    for (int up = 1; up <= 10; ++up) {
      const double pUp = s.p(up);
      s.remove(up);
      for (int c1 = 1; c1 <= 10; ++c1) {
        const double p1 = pUp * s.p(c1);
        s.remove(c1);
        for (int c2 = 1; c2 <= 10; ++c2) {
          if (s.n[c2] == 0) continue;
          const double p = p1 * s.p(c2);
          const solver::ActionValues v = exact.openingValues(c1, c2, up);
          if (!v.legal(Action::HIT)) { // player natural, pushes included
            mass[PLAYER_NATURAL] += p;
            value[PLAYER_NATURAL] += p * v[Action::STAND];
            continue;
          }
          s.remove(c2);
          const double pNat = up == ACE ? s.p(TEN) : up == TEN ? s.p(ACE) : 0.0;
          s.add(c2);
          mass[DEALER_NATURAL] += p * pNat;
          value[DEALER_NATURAL] -= p * pNat;
          // Given no dealer natural, an unconditional value v is worth
          // (v + pNat) / (1 - pNat); weight (1 - pNat) cancels the division.
          auto add = [&](bool canSplit, double w) {
            const Action a = exact.openingAction(c1, c2, up, canSplit);
            const Category c = categoryOf(a);
            mass[c] += w * p * (1.0 - pNat);
            value[c] += w * p * (v[a] + pNat);
          };
          if (c1 == TEN && c2 == TEN && exactRules.splitIdenticalRanksOnly) {
            add(true, qSameTen);
            add(false, 1.0 - qSameTen);
          } else {
            add(true, 1.0);
          }
        }
        s.add(c1);
      }
      s.add(up);
    }
  }

  // ---- Book pass ----
  {
    BlackjackGame game(fresh, util::deriveSeed(options.seed, kBookPass));
    Tally ev;
    std::array<Tally, N_CATEGORIES> share, conditional;
    for (uint64_t i = 0; i < options.rounds; ++i) {
      const Category c = playRound(book, game);
      const double r = roundReward(game, rules.blackjackPayout);
      ev.add(r);
      for (int k = 0; k < N_CATEGORIES; ++k) share[k].add(k == c ? 1.0 : 0.0);
      conditional[c].add(r);
    }
    lines.push_back(line("EV per round, book", bookEv, ev));
    for (int k = 0; k < N_CATEGORIES; ++k) {
      if (k == SURRENDER && !rules.surrender) continue;
      const std::string name = kCategoryName[k];
      lines.push_back(line("P(" + name + ")", mass[k], share[k]));
      // A dealer natural is always exactly -1: nothing to estimate.
      if (k != DEALER_NATURAL) {
        lines.push_back(line("EV | " + name, value[k] / mass[k], conditional[k]));
      }
    }
  }

  // ---- Cut card: the same book, dealt to the rules' penetration ----
  if (options.cutCardRounds > 0) {
    BlackjackGame game(rules, util::deriveSeed(options.seed, kCutCardPass));
    Tally ev;
    for (uint64_t i = 0; i < options.cutCardRounds; ++i) {
      playRound(book, game);
      ev.add(roundReward(game, rules.blackjackPayout));
    }
    const int pct = static_cast<int>(std::lround(rules.penetration * 100));
    lines.push_back(line("EV per round, book, cut card at " + std::to_string(pct) + "%",
                         bookEv, ev));
  }
  return lines;
}

} // namespace training
} // namespace blackjack
