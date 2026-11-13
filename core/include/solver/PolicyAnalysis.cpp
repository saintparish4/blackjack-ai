#include "PolicyAnalysis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace blackjack {
namespace solver {

using ai::Action;

namespace {

constexpr Action kActions[] = {Action::STAND, Action::HIT, Action::DOUBLE,
                               Action::SPLIT, Action::SURRENDER};

HandInfo infoOf(const Decision &d) {
  return HandInfo{d.hand, d.upCard, d.canDouble, d.canSplit, d.canSurrender, d.pairCard != 0};
}

} // namespace

ai::State stateOf(const Decision &d) {
  return AgentPolicy::stateFor(infoOf(d), ai::COUNT_UNTRACKED);
}

std::optional<Action> TabularPolicy::get(const ai::State &state) const {
  auto it = table_.find(state.hash());
  if (it == table_.end()) return std::nullopt;
  return it->second;
}

Action TabularPolicy::choose(const HandInfo &info, const ActionValues *unsplit) const {
  if (auto a = get(AgentPolicy::stateFor(info, ai::COUNT_UNTRACKED))) {
    const bool legal = (*a != Action::DOUBLE || info.canDouble) &&
                       (*a != Action::SPLIT || info.canSplit) &&
                       (*a != Action::SURRENDER || info.canSurrender);
    if (legal) return *a;
  }
  return fallback_->choose(info, unsplit);
}

TotalDependentSearch bestTotalDependentPolicy(const Rules &rules, const Shoe &shoe,
                                              const Policy &start, int maxIterations) {
  TabularPolicy current(start); // empty: plays exactly like start
  TotalDependentSearch result{current, -1e9, {}};

  for (int it = 0; it <= maxIterations; ++it) {
    Solver solver(rules, shoe, current);
    const double ev = solver.expectedValue();
    result.evByIteration.push_back(ev);
    if (ev <= result.ev) break; // no longer improving: keep the best so far
    result.policy = current;
    result.ev = ev;

    struct Score {
      ai::State state;
      std::array<double, 5> q{};
      std::array<bool, 5> legal{true, true, true, true, true};
    };
    std::unordered_map<size_t, Score> scores;
    for (const Decision &d : solver.decisions()) {
      const ActionValues v = solver.decisionValues(d);
      const ai::State s = stateOf(d);
      Score &sc = scores.emplace(s.hash(), Score{s, {}, {true, true, true, true, true}})
                      .first->second;
      for (Action a : kActions) {
        const auto i = static_cast<size_t>(a);
        if (!v.legal(a)) sc.legal[i] = false;
        else sc.q[i] += d.reach * v[a];
      }
    }

    TabularPolicy next(start);
    for (const auto &entry : scores) {
      const Score &sc = entry.second;
      Action best = Action::STAND;
      double bestQ = -1e18;
      for (Action a : kActions) {
        const auto i = static_cast<size_t>(a);
        if (sc.legal[i] && sc.q[i] > bestQ) {
          best = a;
          bestQ = sc.q[i];
        }
      }
      next.set(sc.state, best);
    }
    if (next == current) break;
    current = next;
  }
  return result;
}

RegretReport regretByState(Solver &policy, Solver &reference) {
  RegretReport report;
  report.policyEv = policy.expectedValue();
  report.referenceEv = reference.expectedValue();

  struct Line {
    StateLoss loss;
    std::array<double, 5> referenceReach{};
  };
  std::unordered_map<size_t, Line> lines;
  for (const Decision &d : policy.decisions()) {
    const ActionValues q = reference.decisionValues(d);
    const Action ref = reference.decisionAction(d);
    const double loss = d.reach * (q[ref] - q[d.action]);
    const ai::State s = stateOf(d);
    Line &line = lines[s.hash()];
    line.loss.state = s;
    line.loss.played = d.action;
    line.loss.reach += d.reach;
    line.loss.loss += loss;
    line.referenceReach[static_cast<size_t>(ref)] += d.reach;
    report.itemisedLoss += loss;
  }
  for (auto &entry : lines) {
    Line &l = entry.second;
    const auto most = std::max_element(l.referenceReach.begin(), l.referenceReach.end());
    l.loss.reference = static_cast<Action>(most - l.referenceReach.begin());
    report.byState.push_back(l.loss);
  }
  std::sort(report.byState.begin(), report.byState.end(),
            [](const StateLoss &a, const StateLoss &b) { return a.loss > b.loss; });
  return report;
}

std::string describe(const ai::State &s) {
  static const char *ups[] = {"", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T"};
  std::ostringstream out;
  if (s.canSplit) {
    const int card = s.hasUsableAce ? 1 : s.playerTotal / 2;
    out << ups[card] << "," << ups[card];
  } else {
    out << (s.hasUsableAce ? "soft " : "hard ") << s.playerTotal;
  }
  out << " vs " << ups[s.dealerUpCard] << (s.canDouble && !s.canSplit ? " (2 cards)" : "");
  return out.str();
}

void writeFarmerPolicy(const Policy &policy, const Rules &rules, const std::string &path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  auto code = [](Action a) {
    switch (a) {
    case Action::STAND: return 1;
    case Action::HIT: return 2;
    case Action::DOUBLE: return 3;
    case Action::SPLIT: return 4;
    case Action::SURRENDER: return 5;
    }
    return 1;
  };
  // A composition only needs the right total and softness: every policy
  // written here is total-dependent.
  auto hardHand = [](int t) {
    Composition h;
    if (t <= 11) { h.add(t - 2); h.add(2); }
    else if (t <= 20) { h.add(10); h.add(t - 10); }
    else { h.add(10); h.add(9); h.add(2); }
    return h;
  };
  auto softHand = [](int t) {
    Composition h;
    h.add(ACE);
    if (t > 11) h.add(t - 11);
    return h;
  };
  auto emit = [&](const Composition &h, int up, bool dbl, bool split, bool surrender) {
    const HandInfo info{h, up, dbl, split, surrender, false};
    out << h.total() << ' ' << (h.soft() ? 1 : 0) << ' ' << up << ' ' << (dbl ? 1 : 0)
        << ' ' << (split ? 1 : 0) << ' ' << (surrender ? 1 : 0) << ' '
        << code(policy.choose(info, nullptr)) << '\n';
  };
  for (int up = 1; up <= 10; ++up) {
    for (int t = 4; t <= 21; ++t) {
      const Composition h = hardHand(t);
      emit(h, up, false, false, false); // after a hit, or a split half
      if (t <= 20) emit(h, up, true, false, rules.surrender); // opening
      if (t % 2 == 0 && t <= 20) {                            // opening pair
        Composition pair;
        pair.add(t / 2);
        pair.add(t / 2);
        emit(pair, up, true, true, rules.surrender);
      }
    }
    for (int t = 12; t <= 21; ++t) {
      const Composition h = softHand(t);
      emit(h, up, false, false, false);
      if (t <= 20) emit(h, up, true, false, rules.surrender);
    }
    Composition aces;
    aces.add(ACE);
    aces.add(ACE);
    emit(aces, up, true, true, rules.surrender);
  }
}

} // namespace solver
} // namespace blackjack
