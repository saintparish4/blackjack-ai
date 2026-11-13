// regret: grade learned agents by exact EV lost, not by cells matched.
//
// For a set of saved Q-learning models, prints:
//  1. the reference policies, exactly: composition-dependent optimal, the
//     best total-dependent policy (searched from the book chart), the book;
//  2. each model's exact EV and its shortfall, split into the ceiling of the
//     agent's state representation and the agent's own learning error;
//  3. the bill: which of the agent's States the learning error comes from,
//     averaged over the models, with the check that it sums to the total.
// --export-farmer writes each model's policy for tools/farmer/policy_ev.

#include "ai/QLearningAgent.hpp"
#include "solver/PolicyAnalysis.hpp"
#include "training/Evaluator.hpp"
#include "training/Experiment.hpp"
#include "training/RunConfig.hpp"
#include "util/ArgParser.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace blackjack;
using namespace blackjack::solver;
using ai::Action;

namespace {

std::vector<std::string> splitList(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream in(s);
  std::string item;
  while (std::getline(in, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

const char *name(Action a) {
  static const char *n[] = {"hit", "stand", "double", "split", "surrender"};
  return n[static_cast<size_t>(a)];
}

void printSpread(const char *label, const std::vector<double> &values) {
  const training::Spread s = training::summarize(values);
  std::printf("  %-34s %+.6f%% +/- %.6f%%  (range %+.6f%% .. %+.6f%%)\n", label, s.mean * 100,
              s.ci95 * 100, s.min * 100, s.max * 100);
}

} // namespace

int main(int argc, char *argv[]) {
  util::ArgParser args("regret", "Exact regret of learned agents, itemised by state");
  args.addFlag("models", "m", "Comma-separated model paths (without .qtable/.meta)", "", true);
  args.addFlag("preset", "p", "Rules preset", "vegas-strip");
  args.addFlag("top", "t", "Lines of the bill to print", "15");
  args.addFlag("export-farmer", "", "Write <prefix>-<i>.policy for each model", "");
  args.addBool("help", "h", "Show this help message");
  if (!args.parse(argc, argv)) return 0;

  const GameRules game = training::rulesFromPreset(args.getString("preset"));
  const Rules rules = Rules::fromGame(game);
  const Shoe shoe = Shoe::decks(rules.decks);
  const auto paths = splitList(args.getString("models"));
  const size_t top = static_cast<size_t>(args.getInt("top"));

  // ---- 1. References ----
  OptimalPolicy optimal;
  training::BasicStrategyAgent bookAgent(game);
  AgentPolicy book(bookAgent);
  Solver optimalSolver(rules, optimal);
  Solver bookSolver(rules, book);
  const TotalDependentSearch td = bestTotalDependentPolicy(rules, shoe, book);
  Solver tdSolver(rules, td.policy);
  const double evOptimal = optimalSolver.expectedValue();
  const double evBook = bookSolver.expectedValue();

  std::printf("Rules: %s, exact values off the top of a full shoe\n\n", args.getString("preset").c_str());
  std::printf("References\n");
  std::printf("  %-34s %+.9f%%\n", "composition-dependent optimal", evOptimal * 100);
  std::printf("  %-34s %+.9f%%  (%zu policy-iteration steps from the book)\n",
              "best total-dependent policy (TD*)", td.ev * 100, td.evByIteration.size() - 1);
  std::printf("  %-34s %+.9f%%\n", "book chart", evBook * 100);
  std::printf("  %-34s %.9f%% per round\n", "representation ceiling (CD - TD*)",
              (evOptimal - td.ev) * 100);
  std::printf("  %-34s %.9f%% per round\n", "book's own gap (TD* - book)", (td.ev - evBook) * 100);

  const RegretReport bookBill = regretByState(bookSolver, tdSolver);
  std::printf("\nWhere TD* plays differently from the book\n");
  for (const StateLoss &l : bookBill.byState) {
    if (std::fabs(l.loss) < 1e-12) continue;
    std::printf("  %-26s book %-6s TD* %-6s worth %.7f%% per round\n",
                describe(l.state).c_str(), name(l.played), name(l.reference), l.loss * 100);
  }

  // ---- 2. Each model ----
  std::printf("\nModels\n");
  std::printf("  %-44s %13s %12s %12s %12s\n", "model", "exact EV", "vs CD", "vs TD*", "vs book");
  std::vector<double> evs, toCd, toTd, toBook;
  struct Pooled {
    ai::State state;
    double loss = 0.0, reach = 0.0;
    int models = 0;
    Action reference = Action::STAND;
    std::map<Action, int> played;
  };
  std::map<size_t, Pooled> pooled;

  for (size_t i = 0; i < paths.size(); ++i) {
    ai::QLearningAgent agent;
    agent.load(paths[i]);
    AgentPolicy policy(agent);
    Solver solver(rules, policy);
    const RegretReport bill = regretByState(solver, tdSolver);
    const double ev = bill.policyEv;
    evs.push_back(ev);
    toCd.push_back(evOptimal - ev);
    toTd.push_back(td.ev - ev);
    toBook.push_back(evBook - ev);
    std::printf("  %-44s %+.9f%% %.7f%% %.7f%% %.7f%%\n", paths[i].c_str(), ev * 100,
                (evOptimal - ev) * 100, (td.ev - ev) * 100, (evBook - ev) * 100);
    for (const StateLoss &l : bill.byState) {
      Pooled &p = pooled[l.state.hash()];
      p.state = l.state;
      p.loss += l.loss;
      p.reach += l.reach;
      p.reference = l.reference;
      if (l.loss > 1e-12) {
        ++p.models;
        ++p.played[l.played];
      }
    }
    // An empty default registers no value, so ask has() before getString().
    if (args.has("export-farmer")) {
      const std::string out = args.getString("export-farmer") + "-" + std::to_string(i) + ".policy";
      writeFarmerPolicy(policy, rules, out);
    }
  }
  if (evs.empty()) return 0;
  const double n = static_cast<double>(evs.size());

  std::printf("\nAcross %zu models (t-based 95%% CI over models; every EV is exact, so the\n"
              "interval is training variation alone, with no evaluation noise)\n", evs.size());
  printSpread("exact EV", evs);
  printSpread("shortfall to CD optimal", toCd);
  printSpread("  of which representation ceiling", std::vector<double>(evs.size(), evOptimal - td.ev));
  printSpread("  of which learning error (to TD*)", toTd);
  printSpread("shortfall to the book chart", toBook);

  // ---- 3. The bill ----
  std::vector<Pooled> lines;
  double total = 0.0;
  for (auto &entry : pooled) {
    entry.second.loss /= n;
    entry.second.reach /= n;
    total += entry.second.loss;
    lines.push_back(entry.second);
  }
  std::sort(lines.begin(), lines.end(),
            [](const Pooled &a, const Pooled &b) { return a.loss > b.loss; });
  const double meanLearning = training::summarize(toTd).mean;
  std::printf("\nLearning error by state, mean over models (largest first)\n");
  std::printf("  %-26s %9s %12s %7s %7s  %-10s %s\n", "state", "reach", "loss/round", "share",
              "cum.", "TD* plays", "models wrong (their play)");
  double cumulative = 0.0;
  for (size_t i = 0; i < lines.size() && i < top; ++i) {
    const Pooled &p = lines[i];
    cumulative += p.loss;
    std::string played;
    for (const auto &kv : p.played) {
      if (!played.empty()) played += ", ";
      played += std::to_string(kv.second) + " " + name(kv.first);
    }
    std::printf("  %-26s %9.6f %11.7f%% %6.1f%% %6.1f%%  %-10s %d (%s)\n",
                describe(p.state).c_str(), p.reach, p.loss * 100, 100 * p.loss / meanLearning,
                100 * cumulative / meanLearning, name(p.reference), p.models, played.c_str());
  }
  std::printf("\nCheck: itemised %.12f%% = mean learning error %.12f%% (|diff| %.1e)\n",
              total * 100, meanLearning * 100, std::fabs(total - meanLearning));
  return 0;
}
