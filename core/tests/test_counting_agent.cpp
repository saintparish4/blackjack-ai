// Experiment 005's learners and the score they're judged by.
//
// The point of several of these is an identity, not a behaviour: with its
// options at their defaults CountingAgent *is* experiment 002's learner,
// and a frozen prior *is* deviation learning. If either identity broke, the
// arms would no longer be variations of one thing.

#include "ai/CountingAgent.hpp"
#include "training/IndexDiscovery.hpp"
#include "training/Trainer.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;

namespace {

TabularAgent::Hyperparameters countingParams() {
  TabularAgent::Hyperparameters p;
  p.alphaSchedule = AlphaSchedule::VISIT;
  p.alphaOmega = 1.0;
  p.useCount = true;
  p.epsilonDecay = 0.9999;
  p.epsilonMin = 0.05;
  return p;
}

TrainingConfig quietConfig(uint32_t seed) {
  TrainingConfig c;
  c.gameRules = GameRules::vegasStrip();
  c.seed = seed;
  c.verbose = false;
  c.saveCheckpoints = false;
  return c;
}

State countState(int total, int up, int bucket, bool canDouble = true) {
  State s(total, up, false, false, canDouble);
  s.countBucket = bucket;
  return s;
}

Experience terminal(const State &s, Action a, double reward) {
  return Experience(s, a, reward, State(), true);
}

/** Prefers `deviation` exactly at buckets whose floor of TC is >= threshold. */
class ThresholdAgent : public Agent {
public:
  ThresholdAgent(const IndexPlay &play, int threshold, int minBucketSeen = 1)
      : play_(play), threshold_(threshold), minBucket_(minBucketSeen) {}
  Action chooseAction(const State &, const std::vector<Action> &v, bool) override { return v[0]; }
  void learn(const Experience &) override {}
  double getQValue(const State &s, Action a) const override {
    const bool deviate = trueCountOfBucket(s.countBucket) >= threshold_;
    if (a == play_.atOrAbove) return deviate ? 1.0 : 0.0;
    if (a == play_.below) return deviate ? 0.0 : 1.0;
    return -1.0;
  }
  uint32_t getVisitCount(const State &s, Action) const override {
    return s.countBucket >= minBucket_ ? 1 : 0;
  }
  void save(const std::string &) const override {}
  void load(const std::string &) override {}
  std::string getName() const override { return "threshold"; }

private:
  IndexPlay play_;
  int threshold_;
  int minBucket_;
};

} // namespace

TEST(CountingAgentTest, DefaultOptionsReproduceExperiment002sLearnerBitForBit) {
  auto reference = std::make_shared<QLearningAgent>(countingParams(), 11u);
  auto counting =
      std::make_shared<CountingAgent>(countingParams(), CountingAgent::Options{}, 11u);
  Trainer a(reference, quietConfig(5));
  Trainer b(counting, quietConfig(5));
  for (int i = 0; i < 20000; ++i) {
    a.runEpisode();
    b.runEpisode();
  }
  int compared = 0;
  for (size_t h = 0; h < PolicyTable::TABLE_SIZE; ++h) {
    const State s = PolicyTable::stateFromHash(h);
    for (Action act : {Action::HIT, Action::STAND, Action::DOUBLE, Action::SPLIT}) {
      ASSERT_EQ(reference->getVisitCount(s, act), counting->getVisitCount(s, act)) << h;
      ASSERT_EQ(reference->getQValue(s, act), counting->getQValue(s, act)) << h;
      compared += reference->getVisitCount(s, act) > 0 ? 1 : 0;
    }
  }
  EXPECT_GT(compared, 1000);
}

TEST(CountingAgentTest, FrozenPriorIsDeviationLearningWithTheSamePseudoCount) {
  // A count-blind prior: hard 16 vs T, standing worth -0.5.
  const std::string dir = ::testing::TempDir() + "prior";
  std::filesystem::create_directories(dir);
  QLearningAgent blind(countingParams());
  blind.learn(terminal(State(16, 10, false, false, true), Action::STAND, -0.5));
  blind.save(dir + "/agent");

  CountingAgent::Options o;
  o.prior = CountingAgent::Prior::FROZEN;
  o.priorVisits = 4;
  o.priorModel = dir + "/agent";
  CountingAgent agent(countingParams(), o, 1u);

  // Deviation learning, written out: Q = Q0 + delta, delta shrunk by k.
  const double q0 = -0.5, k = 4;
  double delta = 0.0, n = 0.0;
  const State s = countState(16, 10, 8); // TC +4
  for (double y : {1.0, -1.0, 1.0, 1.0, -1.0, 1.0}) {
    agent.learn(terminal(s, Action::STAND, y));
    n += 1;
    delta += (y - q0 - delta) / (k + n);
    EXPECT_NEAR(agent.getQValue(s, Action::STAND), q0 + delta, 1e-15);
  }
  // And the closed form both are: (k Q0 + sum y) / (k + n).
  EXPECT_NEAR(agent.getQValue(s, Action::STAND), (k * q0 + 2.0) / (k + n), 1e-15);
  // A bucket with no data of its own plays the prior.
  EXPECT_NEAR(agent.getQValue(countState(16, 10, 2), Action::STAND), q0, 1e-15);
}

TEST(CountingAgentTest, PooledPriorLearnsFromEveryBucket) {
  CountingAgent::Options o;
  o.prior = CountingAgent::Prior::POOLED;
  o.priorVisits = 10;
  CountingAgent agent(countingParams(), o, 1u);
  for (int i = 0; i < 50; ++i) {
    agent.learn(terminal(countState(12, 4, 3), Action::STAND, i % 2 ? 1.0 : -0.6));
  }
  // Bucket 9 has never been played; it reads the pooled mean, 0.2.
  EXPECT_NEAR(agent.getQValue(countState(12, 4, 9), Action::STAND), 0.2, 1e-12);
  // Bucket 3 reads its own mean shrunk toward the same 0.2: here equal.
  EXPECT_NEAR(agent.getQValue(countState(12, 4, 3), Action::STAND), 0.2, 1e-12);
}

TEST(CountingAgentTest, LinearRepresentationRecoversALineInTheCount) {
  CountingAgent::Options o;
  o.representation = CountingAgent::Representation::LINEAR;
  CountingAgent agent(countingParams(), o, 1u);
  // y = 0.1 + 0.02 x TC, observed at buckets -3..+6 (TC midpoints).
  for (int b = 1; b <= 10; ++b) {
    const double tc = trueCountOfBucket(b) + 0.5;
    for (int rep = 0; rep < 3; ++rep) {
      agent.learn(terminal(countState(16, 10, b), Action::STAND, 0.1 + 0.02 * tc));
    }
  }
  for (int b = 1; b <= 10; ++b) {
    const double tc = trueCountOfBucket(b) + 0.5;
    EXPECT_NEAR(agent.getQValue(countState(16, 10, b), Action::STAND), 0.1 + 0.02 * tc, 1e-12);
  }
  // The fit is per count-blind state: two cards and three are separate.
  EXPECT_EQ(agent.getVisitCount(countState(16, 10, 4, false), Action::STAND), 0u);
}

TEST(CountingAgentTest, ExploringStartsRandomisesOnlyTheFirstDecisionOfARound) {
  CountingAgent::Options o;
  o.exploration = CountingAgent::Exploration::EXPLORING_STARTS;
  CountingAgent agent(countingParams(), o, 3u);
  const State s = countState(11, 6, 5);
  agent.learn(terminal(s, Action::DOUBLE, 2.0)); // DOUBLE is now the greedy play
  const std::vector<Action> valid{Action::HIT, Action::STAND, Action::DOUBLE};
  std::map<Action, int> first;
  for (int round = 0; round < 3000; ++round) {
    ++first[agent.chooseAction(s, valid, true)];
    EXPECT_EQ(agent.chooseAction(s, valid, true), Action::DOUBLE); // later: greedy
    agent.learn(terminal(countState(20, 6, 5), Action::STAND, 0.0)); // round ends
  }
  for (Action a : valid) EXPECT_NEAR(first[a], 1000, 120) << actionToString(a);
}

TEST(CountingAgentTest, UcbTriesEveryActionBeforeRepeatingOne) {
  CountingAgent::Options o;
  o.exploration = CountingAgent::Exploration::UCB;
  CountingAgent agent(countingParams(), o, 3u);
  const State s = countState(12, 2, 4);
  const std::vector<Action> valid{Action::HIT, Action::STAND, Action::DOUBLE};
  for (Action expected : valid) {
    const Action a = agent.chooseAction(s, valid, true);
    EXPECT_EQ(a, expected);
    agent.learn(terminal(s, a, 0.5)); // all equal, so untried comes first
  }
}

TEST(CountingAgentTest, MonteCarloCreditsASplitWithBothHandsReturns) {
  CountingAgent::Options o;
  o.target = CountingAgent::Target::MONTE_CARLO;
  auto params = countingParams();
  params.discountFactor = 1.0;
  CountingAgent agent(params, o, 1u);
  const State pair(16, 10, false, true, true);
  const State firstHand = countState(16, 10, 0, false);
  const State secondHand = countState(13, 10, 0, false);
  State p = pair;
  p.countBucket = 4;
  State f = firstHand, s2 = secondHand;
  f.countBucket = s2.countBucket = 4;

  Experience split(p, Action::SPLIT, 0.0, f, false);
  split.secondNextState = s2;
  agent.learn(split);
  agent.learn(Experience(f, Action::HIT, 0.0, f, false)); // first hand hits...
  agent.learn(terminal(f, Action::STAND, 1.0));            // ...then wins
  agent.learn(terminal(s2, Action::STAND, -1.0));          // second hand loses
  EXPECT_NEAR(agent.getQValue(p, Action::SPLIT), 0.0, 1e-15); // +1 - 1
  EXPECT_NEAR(agent.getQValue(f, Action::HIT), 1.0, 1e-15);
  EXPECT_NEAR(agent.getQValue(s2, Action::STAND), -1.0, 1e-15);
}

TEST(CountingAgentTest, RejectsOptionsThatBreakTheShrinkageFormula) {
  auto params = countingParams();
  params.alphaOmega = 0.8;
  CountingAgent::Options o;
  o.prior = CountingAgent::Prior::POOLED;
  EXPECT_THROW(CountingAgent(params, o), std::invalid_argument);
  auto blind = countingParams();
  blind.useCount = false;
  EXPECT_THROW(CountingAgent(blind, CountingAgent::Options{}), std::invalid_argument);
}

TEST(IndexDiscoveryTest, FindsAnExactThresholdAndCreditsItWithinOne) {
  for (const IndexPlay &play : illustrious18()) {
    for (int offset : {-2, -1, 0, 1, 2}) {
      const int t = play.index + offset;
      if (t < MIN_TRUE_COUNT || t > MAX_TRUE_COUNT) continue;
      ThresholdAgent agent(play, t);
      const auto learned = learnedIndices(agent);
      const LearnedIndex *li = nullptr;
      for (const auto &l : learned)
        if (l.play.total == play.total && l.play.dealer == play.dealer && l.play.pair == play.pair)
          li = &l;
      ASSERT_NE(li, nullptr);
      EXPECT_EQ(li->lowest, t);
      EXPECT_EQ(li->highest, t);
      EXPECT_EQ(li->mismatches, 0);
      EXPECT_EQ(li->rediscovered, std::abs(offset) <= 1);
    }
  }
}

TEST(IndexDiscoveryTest, DoesNotCreditAThresholdTheDataCannotPinDown) {
  // 15 v T, index +4, but the agent has only seen counts of +5 and +6 (where
  // it stands). Any threshold up to +5 fits; the range, -3..+5, is not "+4".
  const IndexPlay play = illustrious18()[1];
  ASSERT_EQ(play.index, 4);
  ThresholdAgent agent(play, 4, countBucketFor(5));
  const LearnedIndex li = learnedIndices(agent)[1];
  EXPECT_EQ(li.informedBuckets, 2);
  EXPECT_EQ(li.lowest, MIN_TRUE_COUNT);
  EXPECT_EQ(li.highest, 5);
  EXPECT_FALSE(li.rediscovered);
}

TEST(IndexDiscoveryTest, AnAgentThatNeverTriedBothPlaysScoresNothing) {
  QLearningAgent untrained(countingParams());
  const auto learned = learnedIndices(untrained);
  EXPECT_EQ(countRediscovered(learned), 0);
  for (const auto &li : learned) EXPECT_EQ(li.informedBuckets, 0);
}
