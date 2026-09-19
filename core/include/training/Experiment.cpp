#include "Experiment.hpp"
#include "../util/Seed.hpp"
#include "Trainer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <stdexcept>

namespace blackjack {
namespace training {

namespace {

/** Two-sided 95% t critical value. With five seeds the normal 1.96 would
 *  understate the interval by 40%. */
double tCritical95(size_t df) {
  static const double table[] = {
      0.0,   12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
      2.228, 2.201,  2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
      2.086, 2.080,  2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
      2.042};
  if (df == 0) return 0.0;
  if (df <= 30) return table[df];
  return 1.96;
}

struct Metric {
  const char *key;   ///< CSV name
  const char *label; ///< Table label
  double scale;      ///< 100 for percentages
  std::function<double(const SeedResult &)> get;
};

const std::vector<Metric> &bettingMetrics() {
  static const std::vector<Metric> m = {
      {"agent_ev_per_bet", "agent EV / unit bet", 1.0,
       [](const SeedResult &r) { return r.cmp.agent.evPerUnitBet; }},
      {"agent_avg_bet", "agent avg bet", 1.0,
       [](const SeedResult &r) { return r.cmp.agent.avgBet; }},
      {"hilo_ev_per_bet", "Hi-Lo EV / unit bet", 1.0,
       [](const SeedResult &r) { return r.hiLo.evPerUnitBet; }},
      {"hilo_i18_ev_per_bet", "Hi-Lo+I18 EV / unit", 1.0,
       [](const SeedResult &r) { return r.hiLoIndex.evPerUnitBet; }},
      {"hilo_avg_bet", "Hi-Lo avg bet", 1.0,
       [](const SeedResult &r) { return r.hiLo.avgBet; }},
      {"agent_won_per_100", "agent units / 100 rds", 100.0,
       [](const SeedResult &r) { return r.cmp.agent.wonPerRound; }},
      {"hilo_won_per_100", "Hi-Lo units / 100", 100.0,
       [](const SeedResult &r) { return r.hiLo.wonPerRound; }},
      {"hilo_i18_won_per_100", "Hi-Lo+I18 units / 100", 100.0,
       [](const SeedResult &r) { return r.hiLoIndex.wonPerRound; }},
      {"agent_di", "agent DI", 1.0,
       [](const SeedResult &r) { return r.cmp.agent.desirabilityIndex(); }},
      {"hilo_di", "Hi-Lo DI", 1.0,
       [](const SeedResult &r) { return r.hiLo.desirabilityIndex(); }},
      {"hilo_i18_di", "Hi-Lo+I18 DI", 1.0,
       [](const SeedResult &r) { return r.hiLoIndex.desirabilityIndex(); }},
  };
  return m;
}

const std::vector<Metric> &metrics() {
  static const std::vector<Metric> m = {
      {"agent_ev", "agent EV / round", 1.0,
       [](const SeedResult &r) { return r.cmp.agent.avgReward; }},
      {"book_ev", "basic strategy EV", 1.0,
       [](const SeedResult &r) { return r.cmp.book.avgReward; }},
      {"random_ev", "random play EV", 1.0,
       [](const SeedResult &r) { return r.cmp.random.avgReward; }},
      {"gap_recovered", "EV gap recovered %", 100.0,
       [](const SeedResult &r) { return r.cmp.gapRecovered; }},
      {"accuracy", "accuracy %", 100.0,
       [](const SeedResult &r) { return r.audit.accuracy; }},
      {"two_card_accuracy", "two-card accuracy %", 100.0,
       [](const SeedResult &r) { return r.audit.twoCard.accuracy(); }},
      {"multi_card_accuracy", "multi-card accuracy %", 100.0,
       [](const SeedResult &r) { return r.audit.multiCard.accuracy(); }},
      {"pair_accuracy", "pair accuracy %", 100.0,
       [](const SeedResult &r) { return r.audit.pairs.accuracy(); }},
  };
  return m;
}

std::vector<double> column(const std::vector<SeedResult> &results,
                           const Metric &metric) {
  std::vector<double> values;
  values.reserve(results.size());
  for (const auto &r : results) values.push_back(metric.get(r));
  return values;
}

void openForWrite(std::ofstream &file, const std::string &path) {
  std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  file.open(path);
  if (!file) throw std::runtime_error("Cannot open for writing: " + path);
}

} // namespace

Spread summarize(const std::vector<double> &values) {
  Spread s;
  if (values.empty()) return s;
  const double n = static_cast<double>(values.size());
  double sum = 0.0;
  for (double v : values) sum += v;
  s.mean = sum / n;
  s.min = *std::min_element(values.begin(), values.end());
  s.max = *std::max_element(values.begin(), values.end());
  if (values.size() > 1) {
    double ss = 0.0;
    for (double v : values) ss += (v - s.mean) * (v - s.mean);
    s.sd = std::sqrt(ss / (n - 1.0));
    s.ci95 = tCritical95(values.size() - 1) * s.sd / std::sqrt(n);
  }
  return s;
}

SeedResult runSeed(const RunConfig &run, uint32_t seed, size_t evalRounds,
                   const std::string &outDir) {
  TrainingConfig config = run.training;
  config.seed = seed;
  config.verbose = false;
  config.saveCheckpoints = false;
  config.logDir = outDir; // empty = no training log

  auto agent = makeAgent(run, seed);
  Trainer trainer(agent, config);

  SeedResult result;
  result.seed = seed;

  auto start = std::chrono::steady_clock::now();
  TrainingMetrics trained = trainer.trainEpisodes(config.numEpisodes);
  result.trainSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  result.episodes = trained.totalEpisodes;
  result.statesLearned = agent->getStateCount();

  // A dedicated evaluation stream: the final comparison must not depend on how
  // many periodic evaluations happened to run before it.
  Evaluator evaluator(config.gameRules,
                      util::deriveSeed(seed, util::SeedStream::FINAL_EVAL));
  result.cmp = evaluator.compareAgainstBaselines(
      agent.get(), evalRounds,
      util::deriveSeed(seed, util::SeedStream::BASELINE));
  result.audit = evaluator.auditPolicy(agent.get());

  if (config.gameRules.maxBetUnits > 1) {
    result.betting = true;
    HiLoAgent hiLo(config.gameRules, /*indexPlays=*/false);
    HiLoAgent hiLoIndex(config.gameRules, /*indexPlays=*/true);
    result.hiLo = evaluator.evaluate(&hiLo, evalRounds, false);
    result.hiLoIndex = evaluator.evaluate(&hiLoIndex, evalRounds, false);
    for (int b = 1; b <= ai::countBucketFor(ai::MAX_TRUE_COUNT); ++b) {
      const auto i = static_cast<size_t>(b);
      result.learnedEdge[i] = agent->betValue(b);
      result.learnedSamples[i] = agent->betSamples(b);
      ai::BetContext ctx;
      ctx.countBucket = b;
      ctx.maxBet = config.gameRules.maxBetUnits;
      result.learnedBet[i] = agent->chooseBet(ctx, false);
    }
  }

  if (!outDir.empty()) {
    agent->save(outDir + "/agent");
    agent->exportQTable(outDir + "/q_table.csv");
  }
  return result;
}

void writeResultsCsv(const std::string &path,
                     const std::vector<SeedResult> &results) {
  std::ofstream file;
  openForWrite(file, path);
  const bool betting = !results.empty() && results.front().betting;
  file << "seed,episodes,agent_ev,agent_se,book_ev,book_se,random_ev,random_se,"
          "gap_recovered,accuracy,two_card_accuracy,multi_card_accuracy,"
          "pair_accuracy,two_card_scored,multi_card_scored,pair_scored,"
          "unscored_states,states_learned";
  if (betting) {
    for (const char *who : {"agent", "hilo", "hilo_i18"}) {
      file << "," << who << "_ev_per_bet," << who << "_ev_per_bet_se," << who
           << "_avg_bet," << who << "_won_per_round," << who << "_won_sd";
    }
  }
  file << "\n";
  file << std::fixed << std::setprecision(6);
  for (const auto &r : results) {
    file << r.seed << "," << r.episodes << "," << r.cmp.agent.avgReward << ","
         << r.cmp.agent.rewardStdErr << "," << r.cmp.book.avgReward << ","
         << r.cmp.book.rewardStdErr << "," << r.cmp.random.avgReward << ","
         << r.cmp.random.rewardStdErr << "," << r.cmp.gapRecovered << ","
         << r.audit.accuracy << "," << r.audit.twoCard.accuracy() << ","
         << r.audit.multiCard.accuracy() << "," << r.audit.pairs.accuracy() << ","
         << r.audit.twoCard.scored << "," << r.audit.multiCard.scored << ","
         << r.audit.pairs.scored << ","
         << (r.audit.twoCard.untrained + r.audit.multiCard.untrained +
             r.audit.pairs.untrained)
         << "," << r.statesLearned;
    if (betting) {
      for (const EvaluationResult *e : {&r.cmp.agent, &r.hiLo, &r.hiLoIndex}) {
        file << "," << e->evPerUnitBet << "," << e->evPerUnitStdErr << "," << e->avgBet
             << "," << e->wonPerRound << "," << e->wonStdDev;
      }
    }
    file << "\n";
  }
}

void writeCountProfileCsv(const std::string &path,
                          const std::vector<SeedResult> &results) {
  std::ofstream file;
  openForWrite(file, path);
  file << "seed,policy,count_bucket,true_count,rounds,mean_reward,"
          "learned_edge,learned_samples,learned_bet\n";
  file << std::fixed << std::setprecision(6);
  for (const auto &r : results) {
    const std::pair<const char *, const EvaluationResult *> policies[] = {
        {"agent", &r.cmp.agent}, {"basic_strategy", &r.cmp.book},
        {"hilo", &r.hiLo}, {"hilo_i18", &r.hiLoIndex}};
    for (const auto &[name, e] : policies) {
      for (int b = 1; b <= ai::countBucketFor(ai::MAX_TRUE_COUNT); ++b) {
        const auto i = static_cast<size_t>(b);
        const uint64_t n = e->roundsByCount[i];
        const bool agent = std::string(name) == "agent";
        file << r.seed << "," << name << "," << b << "," << ai::trueCountOfBucket(b)
             << "," << n << "," << (n ? e->rewardByCount[i] / static_cast<double>(n) : 0.0)
             << "," << (agent ? r.learnedEdge[i] : 0.0) << ","
             << (agent ? r.learnedSamples[i] : 0) << "," << (agent ? r.learnedBet[i] : 0)
             << "\n";
      }
    }
  }
}

void writeSummaryCsv(const std::string &path,
                     const std::vector<SeedResult> &results) {
  std::ofstream file;
  openForWrite(file, path);
  file << "metric,seeds,mean,sd,ci95,min,max\n";
  file << std::fixed << std::setprecision(6);
  std::vector<const Metric *> all;
  for (const Metric &m : metrics()) all.push_back(&m);
  if (!results.empty() && results.front().betting) {
    for (const Metric &m : bettingMetrics()) all.push_back(&m);
  }
  for (const Metric *m : all) {
    Spread s = summarize(column(results, *m));
    file << m->key << "," << results.size() << "," << s.mean << "," << s.sd << ","
         << s.ci95 << "," << s.min << "," << s.max << "\n";
  }
}

void printResultsTable(const std::vector<SeedResult> &results,
                       std::ostream &out) {
  const auto flags = out.flags();
  const auto precision = out.precision();

  out << std::right << std::setw(6) << "seed" << std::setw(10) << "episodes"
      << std::setw(22) << "agent EV (95% CI)" << std::setw(10) << "book EV"
      << std::setw(10) << "random" << std::setw(8) << "gap %" << std::setw(8)
      << "acc %" << std::setw(8) << "2-card" << std::setw(8) << "multi"
      << std::setw(8) << "pairs" << "\n";
  out << std::string(98, '-') << "\n";
  for (const auto &r : results) {
    out << std::fixed << std::setw(6) << r.seed << std::setw(10) << r.episodes
        << std::setprecision(4) << std::setw(12) << r.cmp.agent.avgReward
        << " +/- " << std::setw(5) << (1.96 * r.cmp.agent.rewardStdErr)
        << std::setw(10) << r.cmp.book.avgReward << std::setw(10)
        << r.cmp.random.avgReward << std::setprecision(1) << std::setw(8)
        << (r.cmp.gapRecovered * 100) << std::setw(8)
        << (r.audit.accuracy * 100) << std::setw(8)
        << (r.audit.twoCard.accuracy() * 100) << std::setw(8)
        << (r.audit.multiCard.accuracy() * 100) << std::setw(8)
        << (r.audit.pairs.accuracy() * 100) << "\n";
  }

  if (!results.empty() && results.front().betting) {
    out << "\nBetting: EV per unit wagered (95% CI) | avg bet | units won per 100 rounds"
           " | desirability index\n";
    const char *names[] = {"agent", "Hi-Lo", "Hi-Lo + I18"};
    for (int c = 0; c < 3; ++c) {
      out << "  " << names[c] << "\n";
      for (const auto &r : results) {
        const EvaluationResult *e =
            c == 0 ? &r.cmp.agent : c == 1 ? &r.hiLo : &r.hiLoIndex;
        out << std::fixed << std::setw(8) << r.seed << std::setprecision(4)
            << std::setw(12) << e->evPerUnitBet << " +/- " << std::setw(6)
            << (1.96 * e->evPerUnitStdErr) << std::setprecision(2) << std::setw(8)
            << e->avgBet << std::setw(9) << (100.0 * e->wonPerRound) << std::setw(8)
            << e->desirabilityIndex() << "\n";
      }
    }
  }

  out << "\nAcross " << results.size() << " seed(s)   "
      << std::setw(10) << "mean" << std::setw(10) << "sd" << std::setw(11)
      << "95% CI" << std::setw(10) << "min" << std::setw(10) << "max" << "\n";
  out << std::string(72, '-') << "\n";
  std::vector<const Metric *> all;
  for (const Metric &m : metrics()) all.push_back(&m);
  if (!results.empty() && results.front().betting) {
    for (const Metric &m : bettingMetrics()) all.push_back(&m);
  }
  for (const Metric *mp : all) {
    const Metric &m = *mp;
    Spread s = summarize(column(results, m));
    const int p = m.scale > 1.0 ? 2 : 4;
    out << std::left << std::setw(22) << m.label << std::right << std::fixed
        << std::setprecision(p) << std::setw(10) << (s.mean * m.scale)
        << std::setw(10) << (s.sd * m.scale) << "  +/-" << std::setw(7)
        << (s.ci95 * m.scale) << std::setw(10) << (s.min * m.scale)
        << std::setw(10) << (s.max * m.scale) << "\n";
  }

  out.flags(flags);
  out.precision(precision);
}

} // namespace training
} // namespace blackjack
