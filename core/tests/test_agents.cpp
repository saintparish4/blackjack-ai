#include "ai/DoubleQLearningAgent.hpp"
#include "ai/ExpectedSarsaAgent.hpp"
#include "ai/QLearningAgent.hpp"
#include "training/RunConfig.hpp"
#include "training/Trainer.hpp"
#include "util/ConfigParser.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;

namespace {

TabularAgent::Hyperparameters params(double alpha, double gamma, double epsilon) {
  TabularAgent::Hyperparameters p;
  p.learningRate = alpha;
  p.discountFactor = gamma;
  p.epsilon = epsilon;
  p.epsilonMin = epsilon;
  p.epsilonDecay = 1.0; // hold ε fixed so expectations are exact
  return p;
}

Experience terminal(const State &s, Action a, double reward) {
  return Experience(s, a, reward, State(), /*done=*/true);
}

const State kS(16, 10, false, false, true);
const State kNext(18, 10, false, false, false);

} // namespace

// === Step-size schedules ===

TEST(AgentTest, VisitScheduleWithOmegaOneIsASampleMean) {
  auto p = params(0.1, 1.0, 0.0);
  p.alphaSchedule = AlphaSchedule::VISIT;
  p.alphaOmega = 1.0;
  QLearningAgent agent(p, 1u);

  // α = 1, 1/2, 1/3: the estimate is the running mean of the targets.
  agent.learn(terminal(kS, Action::HIT, 1.0));
  EXPECT_DOUBLE_EQ(agent.getQValue(kS, Action::HIT), 1.0);
  agent.learn(terminal(kS, Action::HIT, 0.0));
  EXPECT_DOUBLE_EQ(agent.getQValue(kS, Action::HIT), 0.5);
  agent.learn(terminal(kS, Action::HIT, 2.0));
  EXPECT_DOUBLE_EQ(agent.getQValue(kS, Action::HIT), 1.0);
}

TEST(AgentTest, VisitScheduleRespectsTheFloor) {
  auto p = params(0.1, 1.0, 0.0);
  p.alphaSchedule = AlphaSchedule::VISIT;
  p.alphaOmega = 1.0;
  p.alphaMin = 0.5;
  QLearningAgent agent(p, 1u);

  for (int i = 0; i < 10; ++i) agent.learn(terminal(kS, Action::HIT, 0.0));
  // After ten zero-reward updates, α = max(0.5, 1/11) = 0.5.
  agent.learn(terminal(kS, Action::HIT, 1.0));
  EXPECT_DOUBLE_EQ(agent.getQValue(kS, Action::HIT), 0.5);
}

TEST(AgentTest, ConstantScheduleUsesTheLearningRate) {
  QLearningAgent agent(params(0.25, 1.0, 0.0), 1u);
  agent.learn(terminal(kS, Action::STAND, 1.0));
  agent.learn(terminal(kS, Action::STAND, 1.0));
  EXPECT_DOUBLE_EQ(agent.getQValue(kS, Action::STAND), 0.25 + 0.75 * 0.25);
}

TEST(AgentTest, AlphaScheduleParsing) {
  EXPECT_EQ(alphaScheduleFromString("visit"), AlphaSchedule::VISIT);
  EXPECT_EQ(alphaScheduleFromString("constant"), AlphaSchedule::CONSTANT);
  EXPECT_THROW(alphaScheduleFromString("1/n"), std::invalid_argument);
}

// === Bootstrap targets ===

// Next state: HIT worth 1.0, STAND worth 0.0. With ε = 0.2 over two actions the
// ε-greedy policy plays HIT with probability 0.9, so Expected SARSA's target is
// 0.9 while Q-learning's is the max, 1.0.
TEST(AgentTest, ExpectedSarsaBootstrapsFromThePolicysExpectation) {
  const std::vector<Action> next = {Action::HIT, Action::STAND};
  auto p = params(1.0, 1.0, 0.2);

  ExpectedSarsaAgent sarsa(p, 1u);
  QLearningAgent qlearn(p, 1u);
  for (TabularAgent *agent : {static_cast<TabularAgent *>(&sarsa),
                              static_cast<TabularAgent *>(&qlearn)}) {
    agent->learn(terminal(kNext, Action::HIT, 1.0));
    agent->learn(terminal(kNext, Action::STAND, 0.0));
    agent->learn(Experience(kS, Action::HIT, 0.0, kNext, false, next));
  }

  EXPECT_NEAR(sarsa.getQValue(kS, Action::HIT), 0.2 * 0.5 + 0.8 * 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(qlearn.getQValue(kS, Action::HIT), 1.0);
}

// A split moves into two hands, so its value is the sum of both.
TEST(AgentTest, SplitTargetBootstrapsFromBothHands) {
  const std::vector<Action> next = {Action::HIT, Action::STAND};
  const State pair(16, 10, false, /*canSplit=*/true, /*canDouble=*/true);
  const State hand1(18, 10, false, false, false);
  const State hand2(13, 10, false, false, false);

  QLearningAgent agent(params(1.0, 1.0, 0.0), 1u);
  agent.learn(terminal(hand1, Action::STAND, 0.4));
  agent.learn(terminal(hand2, Action::HIT, -0.5));
  agent.learn(terminal(hand2, Action::STAND, -0.6));

  Experience split(pair, Action::SPLIT, 0.0, hand1, false, next);
  split.secondNextState = hand2;
  split.secondValidNextActions = next;
  agent.learn(split);

  EXPECT_DOUBLE_EQ(agent.getQValue(pair, Action::SPLIT), 0.4 + -0.5);
}

TEST(AgentTest, DoubleQCombinedTableTracksTotalUpdates) {
  DoubleQLearningAgent agent(params(1.0, 1.0, 0.0), 3u);
  for (int i = 0; i < 20; ++i) agent.learn(terminal(kS, Action::DOUBLE, 2.0));

  // Whichever half each update landed in, both converge on the reward, and
  // the acting table counts every update.
  EXPECT_DOUBLE_EQ(agent.getQValue(kS, Action::DOUBLE), 2.0);
  EXPECT_EQ(agent.getVisitCount(kS, Action::DOUBLE), 20u);
}

TEST(AgentTest, DoubleQHalfTrainedPairIsNotHalved) {
  // One update lands in exactly one half; the other still reads the 0.0
  // default. The acting value must be that update's value, not its average
  // with an untouched default.
  DoubleQLearningAgent agent(params(1.0, 1.0, 0.0), 5u);
  agent.learn(terminal(kS, Action::HIT, -1.0));
  EXPECT_DOUBLE_EQ(agent.getQValue(kS, Action::HIT), -1.0);
}

// === Interchangeable models ===

TEST(AgentTest, EveryTabularAgentSavesAModelQLearningCanPlay) {
  auto tmp = std::filesystem::temp_directory_path() / "bj_agents_interop";
  std::filesystem::create_directories(tmp);

  TrainingConfig config;
  config.numEpisodes = 20'000;
  config.evalFrequency = 10'000;
  config.evalGames = 200;
  config.verbose = false;
  config.saveCheckpoints = false;
  config.logDir = "";
  config.seed = 9;

  auto p = params(0.1, 1.0, 1.0);
  p.epsilonMin = 0.05;
  p.epsilonDecay = 0.9999;
  std::vector<std::shared_ptr<TabularAgent>> agents = {
      std::make_shared<ExpectedSarsaAgent>(p, 1u),
      std::make_shared<DoubleQLearningAgent>(p, 1u)};

  for (auto &agent : agents) {
    Trainer(agent, config).trainEpisodes(config.numEpisodes);
    const std::string path = (tmp / "model").string();
    agent->save(path);

    QLearningAgent loaded;
    loaded.load(path);
    EXPECT_EQ(loaded.getStateCount(), agent->getStateCount()) << agent->getName();
    for (int total = 5; total <= 20; ++total) {
      for (int dealer = 1; dealer <= 10; ++dealer) {
        State s(total, dealer, false, false, true);
        std::vector<Action> valid = {Action::HIT, Action::STAND, Action::DOUBLE};
        EXPECT_EQ(loaded.chooseAction(s, valid, false),
                  agent->chooseAction(s, valid, false))
            << agent->getName() << " " << s.toString();
      }
    }
  }
  std::filesystem::remove_all(tmp);
}

TEST(AgentTest, RunConfigBuildsEachAgentType) {
  const std::pair<const char *, const char *> cases[] = {
      {"q-learning", "Q-Learning"},
      {"expected-sarsa", "Expected SARSA"},
      {"double-q", "Double Q-Learning"}};
  for (const auto &c : cases) {
    auto dir = std::filesystem::temp_directory_path() / "bj_agents_cfg";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "a.cfg").string();
    {
      std::ofstream f(path);
      f << "agent = " << c.first << "\nalpha_schedule = visit\nalpha_omega = 0.7\n";
    }
    util::ConfigParser cfg;
    cfg.load(path);
    RunConfig run = loadRunConfig(cfg);
    EXPECT_EQ(run.agentParams.alphaSchedule, AlphaSchedule::VISIT);
    EXPECT_DOUBLE_EQ(run.agentParams.alphaOmega, 0.7);
    EXPECT_EQ(makeAgent(run, 1u)->getName(), c.second);
    std::filesystem::remove_all(dir);
  }

  // basic-strategy: book play, and only the bets are learned.
  {
    auto dir = std::filesystem::temp_directory_path() / "bj_agents_book";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "b.cfg").string();
    { std::ofstream f(path); f << "agent = basic-strategy\nlearn_bets = true\n"; }
    util::ConfigParser cfg;
    cfg.load(path);
    auto book = makeAgent(loadRunConfig(cfg), 1u);
    EXPECT_EQ(book->getName(), "Basic strategy + learned bets");
    EXPECT_TRUE(book->learnsBets());
    EXPECT_EQ(book->chooseAction(State(11, 6, false, false, true),
                                 {Action::HIT, Action::STAND, Action::DOUBLE}, true),
              Action::DOUBLE);
    std::filesystem::remove_all(dir);
  }

  util::ConfigParser empty;
  RunConfig defaults = loadRunConfig(empty);
  EXPECT_EQ(defaults.agentType, "q-learning");
}

// === Plain-English explanations (shared by play --beginner and the web demo) ===

#include "ai/Explain.hpp"

TEST(ExplainTest, EveryActionGetsAReasonInEveryKindOfHand) {
  for (Action a : {Action::HIT, Action::STAND, Action::DOUBLE, Action::SPLIT,
                   Action::SURRENDER}) {
    for (const State &s : {State(11, 6, false, false, true), State(17, 4, true, false, true),
                           State(16, 10, false, true, true), State(12, 1, true, true, true)}) {
      EXPECT_FALSE(actionReason(a, s).empty()) << actionToString(a) << " " << s.toString();
    }
  }
}

TEST(ExplainTest, ReasonsFitTheHand) {
  // A soft double shouldn't be explained with the hard-11 line.
  EXPECT_NE(actionReason(Action::DOUBLE, State(17, 4, true, false, true)).find("soft"),
            std::string::npos);
  EXPECT_NE(actionReason(Action::DOUBLE, State(11, 6, false, false, true)).find("11"),
            std::string::npos);
  EXPECT_STREQ(confidenceLabel(0.3), "High");
  EXPECT_STREQ(confidenceLabel(0.01), "Low");
}
