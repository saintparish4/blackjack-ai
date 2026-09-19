#pragma once

#include "Evaluator.hpp"
#include "RunConfig.hpp"
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace blackjack {
namespace training {

/** Everything one seed of an experiment produced. */
struct SeedResult {
  uint32_t seed = 0;
  size_t episodes = 0;       ///< Episodes actually trained (early stopping may cut)
  BaselineComparison cmp;    ///< Agent, book and random EV on this seed's shoes
  PolicyAudit audit;         ///< Cell-match accuracy, split by hand context
  size_t statesLearned = 0;
  double trainSeconds = 0.0; ///< Wall clock; the only nondeterministic field
};

/** Across-seed spread of one metric. */
struct Spread {
  double mean = 0.0;
  double sd = 0.0;     ///< Sample standard deviation (n-1)
  double ci95 = 0.0;   ///< Half-width of a t-based 95% CI on the mean
  double min = 0.0;
  double max = 0.0;
};

/** Summarize values measured once per seed. */
Spread summarize(const std::vector<double> &values);

/**
 * @brief Train and evaluate one seed of a run.
 *
 * The seed fixes the agent's exploration, the training shoe and every
 * evaluation shoe, so the same (config, seed) always yields the same
 * SeedResult apart from trainSeconds.
 *
 * @param evalRounds Rounds per policy for the final EV comparison.
 * @param outDir If non-empty, receives this seed's training log and final model.
 */
SeedResult runSeed(const RunConfig &run, uint32_t seed, size_t evalRounds,
                   const std::string &outDir = "");

/** One row per seed. Wall-clock time is left out so reruns diff clean. */
void writeResultsCsv(const std::string &path,
                     const std::vector<SeedResult> &results);

/** One row per metric: mean, sd, ci95, min, max across seeds. */
void writeSummaryCsv(const std::string &path,
                     const std::vector<SeedResult> &results);

/** Human-readable per-seed table plus the across-seed summary. */
void printResultsTable(const std::vector<SeedResult> &results, std::ostream &out);

} // namespace training
} // namespace blackjack
