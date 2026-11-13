// audit: the engine against the exact solver, for every rules preset.
//
// Each line is an engine estimate next to its exact value and the z-score
// between them. Exits 1 if any off-the-top line misses by more than
// --max-z standard errors; the cut-card line is expected to miss (that gap
// is the cut-card effect) and never fails the run.

#include "training/EngineAudit.hpp"
#include "training/RunConfig.hpp"
#include "util/ArgParser.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace blackjack;

int main(int argc, char *argv[]) {
  util::ArgParser args("audit", "Audit the engine against the exact solver");
  args.addFlag("preset", "p", "One preset, or 'all'", "all");
  args.addFlag("rounds", "r", "Rounds per pass, off the top", "20000000");
  args.addFlag("cut-card-rounds", "", "Rounds dealt to the preset's penetration (0 = skip)", "0");
  args.addFlag("seed", "s", "Seed", "1");
  args.addFlag("max-z", "", "Fail if any off-the-top |z| exceeds this", "4.5");
  args.addBool("help", "h", "Show this help message");
  if (!args.parse(argc, argv)) return 0;

  std::vector<std::string> presets;
  if (args.getString("preset") == "all") {
    presets = {"vegas-strip", "downtown", "atlantic-city", "single-deck"};
  } else {
    presets = {args.getString("preset")};
  }

  training::AuditOptions options;
  options.rounds = std::stoull(args.getString("rounds"));
  options.cutCardRounds = std::stoull(args.getString("cut-card-rounds"));
  options.seed = static_cast<uint32_t>(args.getInt("seed"));
  const double maxZ = args.getDouble("max-z");

  // One thread per preset; each audit is independent and seeded.
  std::vector<std::vector<training::AuditLine>> results(presets.size());
  std::vector<std::thread> threads;
  for (size_t i = 0; i < presets.size(); ++i) {
    threads.emplace_back([&, i] {
      results[i] = training::auditEngine(training::rulesFromPreset(presets[i]), options);
    });
  }
  for (auto &t : threads) t.join();

  bool ok = true;
  for (size_t i = 0; i < presets.size(); ++i) {
    std::printf("\n== %s, %llu rounds per pass, seed %u\n", presets[i].c_str(),
                static_cast<unsigned long long>(options.rounds), options.seed);
    std::printf("%-42s %12s %12s %10s %7s\n", "quantity", "exact", "engine", "+/- se", "z");
    for (const auto &l : results[i]) {
      const bool cutCard = l.name.find("cut card") != std::string::npos;
      const bool miss = !cutCard && std::fabs(l.z()) > maxZ;
      ok = ok && !miss;
      std::printf("%-42s %12.6f %12.6f %10.6f %7.2f%s\n", l.name.c_str(), l.exact,
                  l.mean, l.stdErr, l.z(), miss ? "  <-- MISS" : "");
    }
  }
  std::printf("\n%s\n", ok ? "All off-the-top lines within tolerance."
                           : "Some lines missed: the engine and the solver disagree.");
  return ok ? 0 : 1;
}
