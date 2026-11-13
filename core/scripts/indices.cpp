// indices: can a learner rediscover the Illustrious 18, and how fast?
//
//   indices --config C --theory
//       What the rules alone predict: each play's linear index from effects
//       of removal, the count's playing correlation for it, and how many
//       training episodes telling its two plays apart should take.
//   indices --config C --seeds N [--seed-base B] --checkpoints 1e6,1e7,...
//       Train N seeds of the configured agent and score its learned indices
//       at each checkpoint. Writes <out>/<name>/indices.csv.
//
// Everything printed depends only on the config, seeds and checkpoints.

#include "ai/GameStateConverter.hpp"
#include "solver/CountAnalysis.hpp"
#include "solver/PolicyAnalysis.hpp"
#include "training/Evaluator.hpp"
#include "training/Experiment.hpp"
#include "training/IndexDiscovery.hpp"
#include "training/RunConfig.hpp"
#include "training/Trainer.hpp"
#include "util/ArgParser.hpp"
#include "util/ConfigParser.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace blackjack;
using ai::Action;

namespace {

const char *upName(int up) {
  static const char *n[] = {"", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T"};
  return n[up];
}

std::string playName(const training::IndexPlay &p) {
  std::ostringstream out;
  if (p.pair) out << "T,T";
  else out << p.total;
  out << " v " << upName(p.dealer);
  return out.str();
}

const char *actionName(Action a) {
  static const char *n[] = {"hit", "stand", "double", "split", "surrender"};
  return n[static_cast<size_t>(a)];
}

/// The two-card hand each play is analysed on: T,T, ten-and-X for 12 and up,
/// and X-and-2 for the 9-11 doubles (none of which is a pair).
solver::Composition representative(const training::IndexPlay &p) {
  solver::Composition h;
  if (p.pair) {
    h.add(10);
    h.add(10);
  } else if (p.total >= 12) {
    h.add(10);
    h.add(p.total - 10);
  } else {
    h.add(p.total - 2);
    h.add(2);
  }
  return h;
}

std::vector<size_t> parseCheckpoints(const std::string &s) {
  std::vector<size_t> out;
  std::stringstream in(s);
  std::string item;
  while (std::getline(in, item, ',')) out.push_back(static_cast<size_t>(std::stod(item)));
  return out;
}

/** Share of decisions per bucket, measured by playing the book on the rules. */
std::array<double, 16> bucketShares(const GameRules &rules, uint64_t rounds, uint32_t seed) {
  BlackjackGame game(rules, seed);
  training::BasicStrategyAgent book(rules);
  std::array<double, 16> share{};
  for (uint64_t i = 0; i < rounds; ++i) {
    game.startRound();
    bool first = true;
    while (!game.isRoundComplete()) {
      const Hand &hand = game.getPlayerHand();
      if (first) {
        share[static_cast<size_t>(ai::countBucketFor(game.getTrueCount()))] += 1.0;
        first = false;
      }
      const auto state = ai::GameStateConverter::toAIState(
          hand, game.getDealerHand(true), game.canSplit(), game.canDoubleDown());
      const auto valid = ai::GameStateConverter::getValidActions(
          hand, game.canSplit(), game.canDoubleDown(), game.canSurrender());
      ai::GameStateConverter::executeAction(book.chooseAction(state, valid, false), game);
    }
  }
  double total = 0;
  for (double x : share) total += x;
  for (double &x : share) x /= total;
  return share;
}

void theory(const training::RunConfig &run) {
  const GameRules &rules = run.training.gameRules;
  const solver::Rules exact = solver::Rules::fromGame(rules);
  const solver::Shoe shoe = solver::Shoe::decks(exact.decks);
  training::BasicStrategyAgent bookAgent(rules);
  solver::AgentPolicy book(bookAgent);
  const solver::Tags tags = solver::hiLoTags();

  const auto eor = solver::effectOfRemoval(exact, shoe, book);
  std::printf("Effects of removal on the book's EV, %%/card in one deck (x%d decks):\n ",
              exact.decks);
  for (int r = 1; r <= 10; ++r) std::printf(" %s %+.3f", upName(r), eor[r] * 100 * exact.decks);
  std::printf("\nHi-Lo betting correlation: %.3f (published: 0.97)\n\n",
              solver::weightedCorrelation(eor, tags, shoe));

  // How often each play's opening state comes up, under the book.
  solver::Solver bookSolver(exact, book);
  std::map<size_t, double> reach;
  for (const solver::Decision &d : bookSolver.decisions()) {
    reach[solver::stateOf(d).hash()] += d.reach;
  }
  const auto share = bucketShares(rules, 2'000'000, 12345);
  const double eps = run.agentParams.epsilonMin;

  std::printf("%-8s %5s %7s %6s %9s %8s %9s | predicted episodes to resolve +/-1:\n", "play",
              "index", "linear", "rho", "gain/TC", "reach", "P(rare b)");
  std::printf("%-8s %5s %7s %6s %9s %8s %9s | %12s %12s %12s %12s\n", "", "book", "EoR", "",
              "", "/round", "", "eps table", "ES table", "eps linear", "ES linear");
  for (const training::IndexPlay &p : training::illustrious18()) {
    const solver::Composition h = representative(p);
    const auto est = solver::estimateIndex(exact, shoe, book, h, p.dealer, p.atOrAbove,
                                           p.below, tags);
    ai::State s(p.total, p.dealer, false, p.pair, true);
    const double r = reach[s.hash()];
    // The buckets 1.5 counts either side of the index must read correctly.
    auto bucketShare = [&](int tc) {
      tc = std::max(ai::MIN_TRUE_COUNT, std::min(ai::MAX_TRUE_COUNT, tc));
      return share[static_cast<size_t>(ai::countBucketFor(tc))];
    };
    const double pRare = std::min(bucketShare(p.index + 1), bucketShare(p.index - 2));
    const bool big = p.atOrAbove == Action::DOUBLE || p.atOrAbove == Action::SPLIT;
    const double sigma = big ? 2.0 : 1.0; // SD of one sample of the rarer play
    const int legal = 3 + (p.pair ? 1 : 0) + (rules.surrender ? 1 : 0);
    const double slope = std::fabs(est.slopePerTrueCount);
    const double nBucket = std::pow(2.0 * sigma / (1.5 * slope), 2.0);
    const double nLinear = std::pow(2.0 * sigma / slope, 2.0);
    std::printf("%-8s %+5d %+7.2f %6.2f %8.2f%% %8.5f %9.4f | %12.2e %12.2e %12.2e %12.2e\n",
                playName(p).c_str(), p.index, est.linearIndex, est.correlation, slope * 100, r,
                pRare, nBucket / (r * pRare * eps / legal), nBucket / (r * pRare / legal),
                nLinear / (r * eps / legal), nLinear / (r / legal));
  }
  std::printf("\nPredictions are orders of magnitude: samples needed for a 2-sigma sign at\n"
              "1.5 counts from the index (table) or a 2-sigma index (linear), divided by\n"
              "samples of the rarer play per episode (epsilon_min = %.3f, or 1/k with\n"
              "exploring starts).\n", eps);
}

struct SeedRun {
  uint32_t seed;
  std::vector<std::vector<training::LearnedIndex>> atCheckpoint;
  double seconds = 0;
};

SeedRun trainSeed(const training::RunConfig &run, uint32_t seed,
                  const std::vector<size_t> &checkpoints) {
  training::TrainingConfig config = run.training;
  config.seed = seed;
  config.verbose = false;
  config.saveCheckpoints = false;
  config.logDir.clear();
  auto agent = training::makeAgent(run, seed);
  training::Trainer trainer(agent, config);
  SeedRun out;
  out.seed = seed;
  const auto start = std::chrono::steady_clock::now();
  size_t done = 0;
  for (size_t target : checkpoints) {
    for (; done < target; ++done) trainer.runEpisode();
    out.atCheckpoint.push_back(training::learnedIndices(*agent));
  }
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return out;
}

} // namespace

int main(int argc, char *argv[]) {
  util::ArgParser args("indices", "Rediscovery of the Illustrious 18 by a learned counter");
  args.addFlag("config", "c", "Config file", "", true);
  args.addBool("theory", "", "Print the EoR predictions and exit");
  args.addFlag("seeds", "n", "Seeds to train", "5");
  args.addFlag("seed-base", "", "First seed", "1");
  args.addFlag("checkpoints", "", "Episode counts to score at (comma-separated)",
               "1e6,3e6,1e7,3e7,1e8");
  args.addFlag("threads", "j", "Parallel seeds (0 = hardware)", "0");
  args.addFlag("out", "o", "Output directory", "");
  args.addFlag("name", "", "Run name (subdirectory of --out)", "indices");
  args.addBool("help", "h", "Show this help message");
  if (!args.parse(argc, argv)) return 0;

  util::ConfigParser cfg;
  cfg.load(args.getString("config"));
  const training::RunConfig run = training::loadRunConfig(cfg);

  if (args.getBool("theory")) {
    theory(run);
    return 0;
  }

  const int seeds = args.getInt("seeds");
  const auto base = static_cast<uint32_t>(args.getInt("seed-base"));
  const auto checkpoints = parseCheckpoints(args.getString("checkpoints"));
  unsigned threads = static_cast<unsigned>(args.getInt("threads"));
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());

  std::vector<SeedRun> runs(static_cast<size_t>(seeds));
  for (int startSeed = 0; startSeed < seeds; startSeed += static_cast<int>(threads)) {
    std::vector<std::thread> pool;
    for (int i = startSeed; i < std::min(seeds, startSeed + static_cast<int>(threads)); ++i) {
      pool.emplace_back([&, i] {
        runs[static_cast<size_t>(i)] =
            trainSeed(run, base + static_cast<uint32_t>(i), checkpoints);
      });
    }
    for (auto &t : pool) t.join();
  }

  const auto &plays = training::illustrious18();
  std::printf("Config     : %s\nAgent      : %s\nSeeds      : %u..%u\n\n",
              args.getString("config").c_str(), run.training.agentName.c_str(), base,
              base + static_cast<uint32_t>(seeds) - 1);
  std::printf("%12s  %-26s  %s\n", "episodes", "rediscovered of 17", "per seed");
  for (size_t c = 0; c < checkpoints.size(); ++c) {
    std::vector<double> counts;
    std::string per;
    for (const SeedRun &r : runs) {
      const int k = training::countRediscovered(r.atCheckpoint[c]);
      counts.push_back(k);
      per += std::to_string(k) + " ";
    }
    const training::Spread s = training::summarize(counts);
    std::printf("%12zu  %5.1f +/- %4.1f (%2.0f-%2.0f)       %s\n", checkpoints[c], s.mean,
                s.ci95, s.min, s.max, per.c_str());
  }

  std::printf("\nAt %zu episodes, by play (seeds that rediscovered it; learned range)\n",
              checkpoints.back());
  for (size_t i = 0; i < plays.size(); ++i) {
    int hits = 0;
    std::string ranges;
    for (const SeedRun &r : runs) {
      const auto &li = r.atCheckpoint.back()[i];
      hits += li.rediscovered ? 1 : 0;
      ranges += li.informedBuckets == 0 ? std::string(" none")
                : li.lowest == li.highest
                    ? " " + std::to_string(li.lowest)
                    : " " + std::to_string(li.lowest) + ".." + std::to_string(li.highest);
    }
    std::printf("  %-8s %s->%-6s index %+d  %2d/%d  [%s ]\n", playName(plays[i]).c_str(),
                actionName(plays[i].below), actionName(plays[i].atOrAbove), plays[i].index, hits,
                seeds, ranges.c_str());
  }
  double secs = 0;
  for (const SeedRun &r : runs) secs += r.seconds;
  std::fprintf(stderr, "training: %.0f s across seeds\n", secs);

  // An empty default registers no value, so ask has() before getString().
  if (args.has("out")) {
    const std::filesystem::path dir =
        std::filesystem::path(args.getString("out")) / args.getString("name");
    std::filesystem::create_directories(dir);
    std::ofstream csv(dir / "indices.csv");
    csv << "seed,episodes,play,book_index,informed_buckets,lowest,highest,mismatches,rediscovered\n";
    for (const SeedRun &r : runs) {
      for (size_t c = 0; c < checkpoints.size(); ++c) {
        for (const auto &li : r.atCheckpoint[c]) {
          // Quoted: "T,T v 5" has a comma in it.
          csv << r.seed << ',' << checkpoints[c] << ",\"" << playName(li.play) << "\","
              << li.play.index << ',' << li.informedBuckets << ',' << li.lowest << ','
              << li.highest << ',' << li.mismatches << ',' << (li.rediscovered ? 1 : 0) << '\n';
        }
      }
    }
    std::printf("\nWrote %s\n", (dir / "indices.csv").string().c_str());
  }
  return 0;
}
