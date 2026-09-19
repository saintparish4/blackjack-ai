#include "training/Experiment.hpp"
#include "training/RunConfig.hpp"
#include "util/ArgParser.hpp"
#include "util/ConfigParser.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace blackjack;
using namespace blackjack::training;
using namespace blackjack::util;

// Runs one config over N seeds and reports the spread. Everything on stdout
// and in the CSVs is a pure function of (config, seeds): wall-clock timings go
// to stderr, so two runs of the same command diff clean.
int main(int argc, char *argv[]) {
  ArgParser args("experiment",
                 "Train one config over several seeds and compare EV against "
                 "basic strategy and random play");
  args.addFlag("config", "c", "INI config file (supports 'extends = <path>')", "",
               /*required=*/true);
  args.addFlag("seeds", "n", "Number of seeds", "5");
  args.addFlag("seed-base", "", "First seed; seeds run base, base+1, ...", "1");
  args.addFlag("name", "", "Experiment name (default: config file stem)", "");
  args.addFlag("episodes", "e", "Override episodes per seed", "");
  args.addFlag("eval-rounds", "", "Rounds per policy in the final EV comparison "
               "(default: config eval_rounds, else 200000)", "");
  args.addFlag("out", "o", "Output root; results land in <out>/<name>/",
               "./experiments");
  args.addFlag("threads", "j", "Seeds trained in parallel (default: all cores)", "");
  args.addBool("help", "h", "Show this help message");
  if (!args.parse(argc, argv)) return args.has("help") ? 0 : 1;

  const std::string configPath = args.getString("config");
  ConfigParser cfg;
  RunConfig run;
  size_t numSeeds = 0;
  uint32_t seedBase = 0;
  size_t evalRounds = 0;
  size_t threads = std::max(1u, std::thread::hardware_concurrency());
  try {
    cfg.load(configPath);
    RunOverrides overrides;
    if (args.wasProvided("episodes"))
      overrides.episodes = std::stoul(args.getString("episodes"));
    run = loadRunConfig(cfg, overrides);
    numSeeds = std::stoul(args.getString("seeds"));
    seedBase = static_cast<uint32_t>(std::stoul(args.getString("seed-base")));
    evalRounds = run.training.reportEvalRounds;
    if (args.wasProvided("eval-rounds"))
      evalRounds = std::stoul(args.getString("eval-rounds"));
    if (args.wasProvided("threads")) threads = std::stoul(args.getString("threads"));
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  if (numSeeds == 0) {
    std::cerr << "Error: --seeds must be at least 1\n";
    return 1;
  }

  // Flags registered with an empty default hold no value until passed.
  std::string name = args.has("name") ? args.getString("name") : "";
  if (name.empty()) name = std::filesystem::path(configPath).stem().string();
  const std::filesystem::path outDir =
      std::filesystem::path(args.getString("out")) / name;

  threads = std::max<size_t>(1, std::min(threads, numSeeds));

  std::cout << "Experiment : " << name << "\n"
            << "Config     : " << configPath << "\n"
            << "Agent      : " << run.agentType << "\n"
            << "Rules      : " << run.rulesPreset << " (" << run.training.gameRules.numDecks
            << " decks, " << (run.training.gameRules.dealerHitsSoft17 ? "H17" : "S17")
            << ", surrender " << (run.training.gameRules.surrender ? "on" : "off")
            << ", blackjack pays " << run.training.gameRules.blackjackPayout << ")\n"
            << "Episodes   : " << run.training.numEpisodes << " per seed\n"
            << "Seeds      : " << seedBase << ".." << (seedBase + numSeeds - 1) << "\n"
            << "Eval rounds: " << evalRounds << " per policy per seed\n"
            << "Output     : " << outDir.string() << "\n\n";

  std::vector<SeedResult> results(numSeeds);
  std::atomic<size_t> next{0};
  std::vector<std::string> errors(numSeeds);

  auto worker = [&]() {
    for (size_t i = next++; i < numSeeds; i = next++) {
      const uint32_t seed = seedBase + static_cast<uint32_t>(i);
      try {
        results[i] = runSeed(run, seed, evalRounds,
                             (outDir / ("seed-" + std::to_string(seed))).string());
        std::cerr << "  seed " << seed << " done in " << std::fixed
                  << std::setprecision(1) << results[i].trainSeconds << " s\n";
      } catch (const std::exception &e) {
        errors[i] = e.what();
      }
    }
  };

  auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (size_t t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto &th : pool) th.join();

  for (size_t i = 0; i < numSeeds; ++i) {
    if (!errors[i].empty()) {
      std::cerr << "Error in seed " << (seedBase + i) << ": " << errors[i] << "\n";
      return 1;
    }
  }

  printResultsTable(results, std::cout);

  try {
    writeResultsCsv((outDir / "results.csv").string(), results);
    writeSummaryCsv((outDir / "summary.csv").string(), results);
  } catch (const std::exception &e) {
    std::cerr << "Error writing results: " << e.what() << "\n";
    return 1;
  }
  std::cout << "\nWrote " << (outDir / "results.csv").string() << " and "
            << (outDir / "summary.csv").string() << "\n";

  std::cerr << "Wall time: " << std::fixed << std::setprecision(1)
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
                   .count()
            << " s on " << threads << " thread(s)\n";
  return 0;
}
