#include "ai/GameStateConverter.hpp"
#include "ai/QLearningAgent.hpp"
#include "training/Evaluator.hpp"
#include "training/Experiment.hpp"
#include "training/RunConfig.hpp"
#include "training/Trainer.hpp"
#include "util/ConfigParser.hpp"
#include "util/Seed.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;

namespace {

/// Per-test scratch directory, removed on destruction.
struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const std::string &name)
      : path(std::filesystem::temp_directory_path() / ("bj_lab_" + name)) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() { std::filesystem::remove_all(path); }
  std::string file(const std::string &name) const { return (path / name).string(); }
};

void writeFile(const std::string &path, const std::string &contents) {
  std::ofstream f(path);
  f << contents;
}

TrainingConfig quietConfig(size_t episodes) {
  TrainingConfig config;
  config.numEpisodes = episodes;
  config.evalFrequency = episodes / 4;
  config.evalGames = 500;
  config.verbose = false;
  config.saveCheckpoints = false;
  config.logDir = "";
  config.earlyStoppingPatience = 1000;
  return config;
}

QLearningAgent::Hyperparameters fastDecay() {
  QLearningAgent::Hyperparameters p;
  p.epsilonDecay = 0.9999;
  return p;
}

} // namespace

// === Reproducibility ===

TEST(LabBenchTest, SameSeedGivesIdenticalTrainingRun) {
  TrainingConfig config = quietConfig(20'000);
  config.seed = 7;

  auto a = std::make_shared<QLearningAgent>(fastDecay(), 99u);
  auto b = std::make_shared<QLearningAgent>(fastDecay(), 99u);
  Trainer ta(a, config);
  Trainer tb(b, config);
  ta.trainEpisodes(config.numEpisodes);
  tb.trainEpisodes(config.numEpisodes);

  ASSERT_EQ(ta.getHistory().size(), tb.getHistory().size());
  for (size_t i = 0; i < ta.getHistory().size(); ++i) {
    const TrainingMetrics &x = ta.getHistory()[i];
    const TrainingMetrics &y = tb.getHistory()[i];
    EXPECT_EQ(x.avgReward, y.avgReward) << "evaluation " << i;
    EXPECT_EQ(x.winRate, y.winRate) << "evaluation " << i;
    EXPECT_EQ(x.strategyAccuracy, y.strategyAccuracy) << "evaluation " << i;
    EXPECT_EQ(x.statesLearned, y.statesLearned) << "evaluation " << i;
  }

  // Bit-identical Q-values, not merely close ones.
  for (int total = 5; total <= 20; ++total) {
    for (int dealer = 1; dealer <= 10; ++dealer) {
      State s(total, dealer, false, false, true);
      for (Action act : {Action::HIT, Action::STAND, Action::DOUBLE}) {
        EXPECT_EQ(a->getQValue(s, act), b->getQValue(s, act));
      }
    }
  }
}

TEST(LabBenchTest, DifferentSeedsGiveDifferentRuns) {
  TrainingConfig c1 = quietConfig(20'000);
  TrainingConfig c2 = c1;
  c1.seed = 1;
  c2.seed = 2;
  auto a = std::make_shared<QLearningAgent>(fastDecay(), 1u);
  auto b = std::make_shared<QLearningAgent>(fastDecay(), 2u);
  Trainer ta(a, c1);
  Trainer tb(b, c2);
  ta.trainEpisodes(c1.numEpisodes);
  tb.trainEpisodes(c2.numEpisodes);

  bool anyDifferent = false;
  for (int total = 12; total <= 16 && !anyDifferent; ++total) {
    State s(total, 10, false, false, true);
    anyDifferent = a->getQValue(s, Action::HIT) != b->getQValue(s, Action::HIT);
  }
  EXPECT_TRUE(anyDifferent);
}

TEST(LabBenchTest, SeededEvaluatorIsDeterministicAcrossInstances) {
  BasicStrategyAgent book;
  Evaluator e1(GameRules::vegasStrip(), 5u);
  Evaluator e2(GameRules::vegasStrip(), 5u);
  EvaluationResult r1 = e1.evaluate(&book, 5'000, false);
  EvaluationResult r2 = e2.evaluate(&book, 5'000, false);
  EXPECT_EQ(r1.avgReward, r2.avgReward);
  EXPECT_EQ(r1.wins, r2.wins);
  EXPECT_EQ(r1.handsPlayed, r2.handsPlayed);

  // Successive calls on one evaluator deal fresh shoes rather than replaying.
  EvaluationResult r3 = e1.evaluate(&book, 5'000, false);
  EXPECT_NE(r1.avgReward, r3.avgReward);
}

TEST(LabBenchTest, ExperimentSeedIsReproducible) {
  util::ConfigParser cfg;
  RunOverrides o;
  o.episodes = 20'000;
  RunConfig run = loadRunConfig(cfg, o);

  SeedResult x = runSeed(run, 3, 20'000);
  SeedResult y = runSeed(run, 3, 20'000);
  EXPECT_EQ(x.episodes, y.episodes);
  EXPECT_EQ(x.cmp.agent.avgReward, y.cmp.agent.avgReward);
  EXPECT_EQ(x.cmp.book.avgReward, y.cmp.book.avgReward);
  EXPECT_EQ(x.cmp.random.avgReward, y.cmp.random.avgReward);
  EXPECT_EQ(x.audit.accuracy, y.audit.accuracy);
  EXPECT_EQ(x.statesLearned, y.statesLearned);
}

TEST(LabBenchTest, DerivedSeedStreamsDiffer) {
  using util::deriveSeed;
  using util::SeedStream;
  EXPECT_NE(deriveSeed(1, SeedStream::AGENT), deriveSeed(1, SeedStream::TRAIN_DEAL));
  EXPECT_NE(deriveSeed(1, SeedStream::AGENT), deriveSeed(2, SeedStream::AGENT));
  EXPECT_EQ(deriveSeed(9, SeedStream::EVAL_DEAL), deriveSeed(9, SeedStream::EVAL_DEAL));
}

// === Early stopping on a signal that actually moves ===

// A fixed policy's audit accuracy never changes, so with patience P training
// must stop at exactly evaluation P+1: the first sets the best, the next P do
// not beat it. Under the old win-rate signal the stopping point was random,
// because a 43% win rate over a few hundred rounds wobbles by several points
// and every lucky wobble reset the patience counter.
TEST(LabBenchTest, EarlyStoppingOnAccuracyStopsWhenThePolicyStopsChanging) {
  TrainingConfig config = quietConfig(1'000'000);
  config.evalFrequency = 10;
  config.evalGames = 200;
  config.earlyStoppingPatience = 3;
  config.earlyStoppingMetric = EarlyStopMetric::ACCURACY;

  auto book = std::make_shared<BasicStrategyAgent>();
  Trainer trainer(book, config);
  TrainingMetrics m = trainer.trainEpisodes(config.numEpisodes);

  EXPECT_EQ(m.totalEpisodes, 40u);
  EXPECT_DOUBLE_EQ(m.strategyAccuracy, 1.0);
}

TEST(LabBenchTest, EarlyStoppingOnEvUsesTheRewardSignal) {
  TrainingConfig config = quietConfig(1'000'000);
  config.evalFrequency = 10;
  config.evalGames = 200;
  config.earlyStoppingPatience = 2;
  config.earlyStoppingMetric = EarlyStopMetric::EV;
  // No finite sample mean can beat the best by 10 units, so the stop point is
  // exact even though EV itself is noisy.
  config.minImprovement = 10.0;

  auto book = std::make_shared<BasicStrategyAgent>();
  Trainer trainer(book, config);
  EXPECT_EQ(trainer.trainEpisodes(config.numEpisodes).totalEpisodes, 30u);
}

TEST(LabBenchTest, EarlyStopMetricParsing) {
  EXPECT_EQ(earlyStopMetricFromString("accuracy"), EarlyStopMetric::ACCURACY);
  EXPECT_EQ(earlyStopMetricFromString("ev"), EarlyStopMetric::EV);
  EXPECT_THROW(earlyStopMetricFromString("win_rate"), std::invalid_argument);
}

// === Blackjack payout ===

TEST(LabBenchTest, OutcomeRewardHonoursBlackjackPayout) {
  EXPECT_DOUBLE_EQ(
      GameStateConverter::outcomeToReward(Outcome::PLAYER_BLACKJACK, false, 1.2),
      1.2);
  EXPECT_DOUBLE_EQ(
      GameStateConverter::outcomeToReward(Outcome::PLAYER_WIN, false, 1.2), 1.0);
}

// Same seed, same fixed policy: the two runs deal identical rounds, so the EV
// difference must be exactly 0.3 per natural. A 6:5 table used to train and
// score as if it paid 3:2.
TEST(LabBenchTest, SixToFivePayoutReachesTheEvaluator) {
  GameRules threeTwo = GameRules::vegasStrip();
  GameRules sixFive = threeTwo;
  sixFive.blackjackPayout = 1.2;

  BasicStrategyAgent book;
  const size_t rounds = 20'000;
  EvaluationResult r32 = Evaluator(threeTwo, 11u).evaluate(&book, rounds, false);
  EvaluationResult r65 = Evaluator(sixFive, 11u).evaluate(&book, rounds, false);

  ASSERT_EQ(r32.blackjacks, r65.blackjacks);
  ASSERT_GT(r32.blackjacks, 0u);
  EXPECT_NEAR(r32.avgReward - r65.avgReward,
              0.3 * static_cast<double>(r32.blackjacks) / rounds, 1e-9);
}

// === Config loading ===

TEST(LabBenchTest, ConfigExtendsLoadsBaseThenOverrides) {
  TempDir dir("extends");
  writeFile(dir.file("base.cfg"), "learning_rate = 0.05\nepisodes = 100\n");
  // extends appears after a key on purpose: the child still wins.
  writeFile(dir.file("child.cfg"),
            "episodes = 200\nextends = base.cfg\nseed = 4\n");

  util::ConfigParser cfg;
  cfg.load(dir.file("child.cfg"));
  EXPECT_EQ(cfg.getInt("episodes"), 200);
  EXPECT_DOUBLE_EQ(cfg.getDouble("learning_rate"), 0.05);
  EXPECT_EQ(cfg.getInt("seed"), 4);
  EXPECT_FALSE(cfg.has("extends"));
}

TEST(LabBenchTest, ConfigExtendsCycleThrows) {
  TempDir dir("cycle");
  writeFile(dir.file("a.cfg"), "extends = b.cfg\n");
  writeFile(dir.file("b.cfg"), "extends = a.cfg\n");
  util::ConfigParser cfg;
  EXPECT_THROW(cfg.load(dir.file("a.cfg")), std::runtime_error);
}

TEST(LabBenchTest, RunConfigPrecedenceIsCliThenConfigThenDefault) {
  TempDir dir("precedence");
  writeFile(dir.file("run.cfg"),
            "episodes = 1234\nseed = 8\nrules_preset = downtown\n"
            "blackjack_payout = 1.2\nearly_stopping_metric = ev\n");
  util::ConfigParser cfg;
  cfg.load(dir.file("run.cfg"));

  RunConfig fromConfig = loadRunConfig(cfg);
  EXPECT_EQ(fromConfig.training.numEpisodes, 1234u);
  EXPECT_EQ(fromConfig.training.seed, std::optional<uint32_t>(8));
  EXPECT_EQ(fromConfig.rulesPreset, "downtown");
  EXPECT_TRUE(fromConfig.training.gameRules.surrender);
  EXPECT_DOUBLE_EQ(fromConfig.training.gameRules.blackjackPayout, 1.2);
  EXPECT_EQ(fromConfig.training.earlyStoppingMetric, EarlyStopMetric::EV);
  // Not in the file: built-in default.
  EXPECT_DOUBLE_EQ(fromConfig.agentParams.learningRate, 0.1);

  RunOverrides cli;
  cli.episodes = 99;
  cli.seed = 3;
  cli.rulesPreset = "vegas-strip";
  RunConfig fromCli = loadRunConfig(cfg, cli);
  EXPECT_EQ(fromCli.training.numEpisodes, 99u);
  EXPECT_EQ(fromCli.training.seed, std::optional<uint32_t>(3));
  EXPECT_FALSE(fromCli.training.gameRules.surrender);
  // A per-field rule override still applies on top of the CLI's preset.
  EXPECT_DOUBLE_EQ(fromCli.training.gameRules.blackjackPayout, 1.2);
}

TEST(LabBenchTest, UnknownPresetIsAnErrorNotASilentFallback) {
  EXPECT_THROW(rulesFromPreset("vegas-stirp"), std::invalid_argument);
}

// === Formatting and summaries ===

TEST(LabBenchTest, SavedMetaKeepsFullEpsilonDecay) {
  TempDir dir("meta");
  QLearningAgent::Hyperparameters p;
  p.epsilonDecay = 0.9999985;
  QLearningAgent agent(p, 1u);
  agent.save(dir.file("agent"));

  std::ifstream meta(dir.file("agent.meta"));
  std::stringstream contents;
  contents << meta.rdbuf();
  EXPECT_NE(contents.str().find("epsilon_decay: 0.9999985"), std::string::npos)
      << contents.str();
}

TEST(LabBenchTest, SummarizeUsesSampleSdAndTInterval) {
  Spread s = summarize({1.0, 2.0, 3.0, 4.0, 5.0});
  EXPECT_DOUBLE_EQ(s.mean, 3.0);
  EXPECT_NEAR(s.sd, std::sqrt(2.5), 1e-12);
  // t(4) = 2.776, not the normal 1.96
  EXPECT_NEAR(s.ci95, 2.776 * std::sqrt(2.5) / std::sqrt(5.0), 1e-9);
  EXPECT_DOUBLE_EQ(s.min, 1.0);
  EXPECT_DOUBLE_EQ(s.max, 5.0);

  EXPECT_DOUBLE_EQ(summarize({4.0}).ci95, 0.0);
}

TEST(LabBenchTest, TrainingReportPrintsFractionalDuration) {
  TempDir dir("report");
  TrainingConfig config = quietConfig(2'000);
  config.reportDir = dir.path.string();
  config.reportEvalRounds = 2'000;
  config.epsilonDecay = 0.9999985;
  config.seed = 1;

  Trainer trainer(std::make_shared<QLearningAgent>(fastDecay(), 1u), config);
  trainer.train();

  std::ifstream report(dir.file("training_report.txt"));
  std::stringstream contents;
  contents << report.rdbuf();
  const std::string text = contents.str();
  EXPECT_EQ(text.find(" seconds\n"), std::string::npos);
  const size_t line = text.find("Training time           : ");
  ASSERT_NE(line, std::string::npos);
  // "0.04 s", never a truncated whole-second count
  EXPECT_NE(text.substr(line, 40).find('.'), std::string::npos);
  EXPECT_NE(text.find("0.9999985"), std::string::npos);
  EXPECT_NE(text.find("Seed                    : 1"), std::string::npos);
}
