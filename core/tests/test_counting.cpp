#include "ai/PolicyTable.hpp"
#include "ai/QLearningAgent.hpp"
#include "game/BlackjackGame.hpp"
#include "training/Evaluator.hpp"
#include "training/Trainer.hpp"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace blackjack;
using namespace blackjack::ai;
using namespace blackjack::training;

namespace {

int hiLoSum(const Hand &hand) {
  int sum = 0;
  for (const Card &c : hand.getCards()) sum += c.hiLo();
  return sum;
}

/// Always bets the table maximum and plays basic strategy.
class MaxBettor : public BasicStrategyAgent {
public:
  using BasicStrategyAgent::BasicStrategyAgent;
  int chooseBet(const BetContext &ctx, bool) override { return ctx.maxBet; }
};

std::filesystem::path scratch(const std::string &name) {
  auto dir = std::filesystem::temp_directory_path() / ("bj_count_" + name);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

} // namespace

// === Counting in the engine ===

TEST(CountingTest, HiLoTagsBalanceOverADeck) {
  int sum = 0;
  for (int suit = 0; suit < 4; ++suit)
    for (int rank = 1; rank <= 13; ++rank)
      sum += Card(static_cast<Rank>(rank), static_cast<Suit>(suit)).hiLo();
  EXPECT_EQ(sum, 0);
  EXPECT_EQ(Card(Rank::FIVE, Suit::HEARTS).hiLo(), 1);
  EXPECT_EQ(Card(Rank::EIGHT, Suit::HEARTS).hiLo(), 0);
  EXPECT_EQ(Card(Rank::KING, Suit::HEARTS).hiLo(), -1);
  EXPECT_EQ(Card(Rank::ACE, Suit::HEARTS).hiLo(), -1);
}

// The running count must move by exactly the cards the player has seen: while
// the hand is live that excludes the dealer's hole card, and once the round is
// over it includes every card on the table. Counting the hole card early would
// hand a counter information no player has.
TEST(CountingTest, RunningCountTracksOnlyVisibleCards) {
  BlackjackGame game(GameRules::vegasStrip(), 31u);
  int checkedLive = 0;
  for (int round = 0; round < 3000; ++round) {
    game.reshuffleIfDue();
    const int before = game.getRunningCount();
    game.startRound();

    if (!game.isRoundComplete()) {
      const Hand dealerUp = game.getDealerHand(/*hideHoleCard=*/true);
      EXPECT_EQ(game.getRunningCount() - before,
                hiLoSum(game.getPlayerHand()) + hiLoSum(dealerUp));
      ++checkedLive;
      // Hit a stiff hand once to exercise mid-round cards, then stand out.
      if (game.getPlayerHand().getTotal() < 12) game.hit();
      while (!game.isRoundComplete()) game.stand();
    }

    int table = hiLoSum(game.getDealerHand(false));
    for (const Hand &h : game.getPlayerHands()) table += hiLoSum(h);
    EXPECT_EQ(game.getRunningCount() - before, table) << "round " << round;
  }
  EXPECT_GT(checkedLive, 1000);
}

TEST(CountingTest, TrueCountDividesByUnseenDecks) {
  BlackjackGame game(GameRules::vegasStrip(), 5u);
  for (int i = 0; i < 40; ++i) {
    game.startRound();
    while (!game.isRoundComplete()) game.stand();
  }
  game.startRound();
  if (!game.isRoundComplete()) {
    // The hole card is dealt but unseen, so it still counts as unseen.
    EXPECT_NEAR(game.getTrueCount(),
                game.getRunningCount() / game.getDecksRemaining(), 1e-12);
    EXPECT_GT(game.getDecksRemaining(), 0.0);
  }
}

TEST(CountingTest, ReshuffleResetsTheCount) {
  GameRules rules = GameRules::singleDeck();
  rules.penetration = 0.2;
  BlackjackGame game(rules, 9u);
  bool sawReset = false;
  for (int i = 0; i < 200 && !sawReset; ++i) {
    if (game.reshuffleIfDue()) {
      EXPECT_EQ(game.getRunningCount(), 0);
      EXPECT_DOUBLE_EQ(game.getDecksRemaining(), 1.0);
      sawReset = true;
    }
    game.startRound();
    while (!game.isRoundComplete()) game.stand();
  }
  EXPECT_TRUE(sawReset);
}

// === State ===

TEST(CountingTest, CountBucketsClampAndRoundDown) {
  EXPECT_EQ(countBucketFor(-10.0), 1);
  EXPECT_EQ(countBucketFor(-2.5), 1); // floor(-2.5) = -3
  EXPECT_EQ(countBucketFor(-0.1), NEUTRAL_COUNT_BUCKET - 1);
  EXPECT_EQ(countBucketFor(0.0), NEUTRAL_COUNT_BUCKET);
  EXPECT_EQ(countBucketFor(0.99), NEUTRAL_COUNT_BUCKET);
  EXPECT_EQ(countBucketFor(1.0), NEUTRAL_COUNT_BUCKET + 1);
  EXPECT_EQ(countBucketFor(6.2), 10);
  EXPECT_EQ(countBucketFor(40.0), 10);
  EXPECT_EQ(trueCountOfBucket(NEUTRAL_COUNT_BUCKET), 0);
  EXPECT_EQ(trueCountOfBucket(1), MIN_TRUE_COUNT);
}

TEST(CountingTest, CountBucketIsPartOfTheKeyAndRoundTrips) {
  State blind(16, 10, false, false, true);
  State counted = blind;
  counted.countBucket = 7;
  EXPECT_NE(blind.hash(), counted.hash());
  EXPECT_NE(blind, counted);
  // Non-counting states keep the hash they had before counting existed.
  EXPECT_LT(blind.hash(), 4096u);

  for (size_t h : {counted.hash(), blind.hash()}) {
    EXPECT_EQ(PolicyTable::stateFromHash(h).hash(), h);
  }
  EXPECT_EQ(PolicyTable::stateFromHash(counted.hash()).countBucket, 7);
}

TEST(CountingTest, QTableV3KeepsCountBucketsAndV2StillLoads) {
  auto dir = scratch("qtable");
  PolicyTable table;
  State counted(13, 2, false, false, true);
  counted.countBucket = 2;
  State blind(13, 2, false, false, true);
  table.set(counted, Action::HIT, 0.25);
  table.set(blind, Action::STAND, -0.5);
  table.saveToBinary((dir / "v3.qtable").string());

  PolicyTable loaded;
  loaded.loadFromBinary((dir / "v3.qtable").string());
  EXPECT_DOUBLE_EQ(loaded.get(counted, Action::HIT), 0.25);
  EXPECT_DOUBLE_EQ(loaded.get(blind, Action::STAND), -0.5);
  EXPECT_EQ(loaded.getVisitCount(counted, Action::STAND), 0u);
  EXPECT_EQ(loaded.size(), 2u);

  // A v2 file, byte for byte as the previous format wrote it.
  {
    std::ofstream f(dir / "v2.qtable", std::ios::binary);
    uint32_t version = 2;
    uint64_t n = 1;
    int total = 16, dealer = 10;
    bool soft = false, split = false, dbl = true;
    double q[5] = {-0.4, -0.5, -0.9, 0.0, 0.0};
    uint32_t visits[5] = {10, 12, 3, 0, 0};
    f.write(reinterpret_cast<char *>(&version), sizeof version);
    f.write(reinterpret_cast<char *>(&n), sizeof n);
    f.write(reinterpret_cast<char *>(&total), sizeof total);
    f.write(reinterpret_cast<char *>(&dealer), sizeof dealer);
    f.write(reinterpret_cast<char *>(&soft), sizeof soft);
    f.write(reinterpret_cast<char *>(&split), sizeof split);
    f.write(reinterpret_cast<char *>(&dbl), sizeof dbl);
    f.write(reinterpret_cast<char *>(q), sizeof q);
    f.write(reinterpret_cast<char *>(visits), sizeof visits);
  }
  PolicyTable old;
  old.loadFromBinary((dir / "v2.qtable").string());
  EXPECT_DOUBLE_EQ(old.get(State(16, 10, false, false, true), Action::HIT), -0.4);
  EXPECT_EQ(old.getVisitCount(State(16, 10, false, false, true), Action::STAND), 12u);
  std::filesystem::remove_all(dir);
}

// === Hi-Lo baseline ===

TEST(CountingTest, HiLoRampAndIndexPlays) {
  EXPECT_EQ(HiLoAgent::rampUnits(-3, 8), 1);
  EXPECT_EQ(HiLoAgent::rampUnits(1, 8), 1);
  EXPECT_EQ(HiLoAgent::rampUnits(2, 8), 2);
  EXPECT_EQ(HiLoAgent::rampUnits(3, 8), 4);
  EXPECT_EQ(HiLoAgent::rampUnits(4, 8), 6);
  EXPECT_EQ(HiLoAgent::rampUnits(6, 8), 8);
  EXPECT_EQ(HiLoAgent::rampUnits(6, 4), 4); // capped by the table

  auto play = [](int total, int dealer, bool canDouble, int tc) {
    return HiLoAgent::indexPlay(State(total, dealer, false, false, canDouble), tc);
  };
  EXPECT_EQ(play(16, 10, true, 0), Action::STAND);  // the famous one
  EXPECT_EQ(play(16, 10, true, -1), Action::HIT);
  EXPECT_EQ(play(11, 1, true, 1), Action::DOUBLE);
  EXPECT_EQ(play(11, 1, false, 1), Action::HIT);    // can't double any more
  EXPECT_EQ(play(12, 5, true, -2), Action::STAND);
  EXPECT_EQ(play(12, 5, true, -3), Action::HIT);
  EXPECT_FALSE(play(14, 10, true, 3).has_value());
  EXPECT_FALSE(HiLoAgent::indexPlay(State(18, 6, true, false, true), 4).has_value());
  EXPECT_EQ(HiLoAgent::indexPlay(State(20, 6, false, true, true), 4), Action::SPLIT);
  EXPECT_EQ(HiLoAgent::indexPlay(State(20, 6, false, true, true), 3), Action::STAND);
  EXPECT_FALSE(HiLoAgent::indexPlay(State(16, 10, false, true, true), 0).has_value());
}

// === Betting in the evaluator ===

TEST(CountingTest, FlatBettorsEdgeEqualsItsEvPerRound) {
  GameRules rules = GameRules::vegasStrip();
  rules.maxBetUnits = 8;
  Evaluator ev(rules, 3u);
  BasicStrategyAgent book(rules);
  EvaluationResult r = ev.evaluate(&book, 20'000, false);
  EXPECT_DOUBLE_EQ(r.avgBet, 1.0);
  EXPECT_NEAR(r.evPerUnitBet, r.avgReward, 1e-12);
  EXPECT_NEAR(r.evPerUnitStdErr, r.rewardStdErr, 1e-9);
}

TEST(CountingTest, BetSizeScalesWinningsNotEdge) {
  GameRules rules = GameRules::vegasStrip();
  rules.maxBetUnits = 8;
  MaxBettor bettor(rules);
  EvaluationResult r = Evaluator(rules, 3u).evaluate(&bettor, 20'000, false);
  EXPECT_DOUBLE_EQ(r.avgBet, 8.0);
  EXPECT_NEAR(r.evPerUnitBet, r.avgReward, 1e-12);
  EXPECT_NEAR(r.wonPerRound, 8.0 * r.avgReward, 1e-9);
}

// Betting the same amount on every round scales the winnings and their spread
// together; a risk-adjusted score must not reward that.
TEST(CountingTest, DesirabilityIndexIgnoresUnitSize) {
  GameRules rules = GameRules::vegasStrip();
  rules.maxBetUnits = 8;
  BasicStrategyAgent flat(rules);
  MaxBettor big(rules);
  EvaluationResult a = Evaluator(rules, 21u).evaluate(&flat, 30'000, false);
  EvaluationResult b = Evaluator(rules, 21u).evaluate(&big, 30'000, false);
  EXPECT_NEAR(b.wonStdDev, 8.0 * a.wonStdDev, 1e-9);
  EXPECT_NEAR(a.desirabilityIndex(), b.desirabilityIndex(), 1e-9);
  EXPECT_NEAR(a.wonStdDev, a.rewardStdErr * std::sqrt(30'000.0), 1e-9);
}

TEST(CountingTest, CountProfileCoversEveryRound) {
  GameRules rules = GameRules::vegasStrip();
  BasicStrategyAgent book(rules);
  EvaluationResult r = Evaluator(rules, 8u).evaluate(&book, 50'000, false);
  uint64_t rounds = 0;
  for (uint64_t n : r.roundsByCount) rounds += n;
  EXPECT_EQ(rounds, 50'000u);
  EXPECT_EQ(r.roundsByCount[COUNT_UNTRACKED], 0u);
  EXPECT_GT(r.roundsByCount[NEUTRAL_COUNT_BUCKET], r.roundsByCount[10]);
}

// === Learned bets ===

TEST(CountingTest, LearnedBetsFollowTheSignOfTheMeasuredEdge) {
  TabularAgent::Hyperparameters p;
  p.learnBets = true;
  QLearningAgent agent(p, 1u);
  BetContext hot, cold, rare;
  hot.countBucket = 8;
  cold.countBucket = 3;
  rare.countBucket = 10;
  for (BetContext *c : {&hot, &cold, &rare}) c->maxBet = 8;

  for (uint64_t i = 0; i < TabularAgent::MIN_BET_SAMPLES; ++i) {
    agent.learnBet(hot, (i % 2) ? 1.2 : -1.0);  // mean +0.1
    agent.learnBet(cold, (i % 2) ? 1.0 : -1.1); // mean -0.05
  }
  agent.learnBet(rare, 5.0); // one lucky sample isn't evidence

  EXPECT_NEAR(agent.betValue(8), 0.1, 1e-12);
  EXPECT_EQ(agent.chooseBet(hot), 8);
  EXPECT_EQ(agent.chooseBet(cold), 1);
  EXPECT_EQ(agent.chooseBet(rare), 1);

  // Proportional sizing: one unit per quarter point of edge, capped.
  TabularAgent::Hyperparameters kelly = p;
  kelly.betProportional = true;
  QLearningAgent sizer(kelly, 1u);
  BetContext mild = hot;
  mild.countBucket = 6;
  for (uint64_t i = 0; i < TabularAgent::MIN_BET_SAMPLES; ++i) {
    sizer.learnBet(mild, (i % 2) ? 1.0 : -0.99); // mean +0.005 -> 2 units
    sizer.learnBet(hot, (i % 2) ? 1.2 : -1.0);   // mean +0.1 -> capped at 8
    sizer.learnBet(cold, (i % 2) ? 1.0 : -1.1);
  }
  EXPECT_EQ(sizer.chooseBet(mild), 2);
  EXPECT_EQ(sizer.chooseBet(hot), 8);
  EXPECT_EQ(sizer.chooseBet(cold), 1);

  // The bet table survives a save/load round trip, as does the flag.
  auto dir = scratch("bets");
  agent.save((dir / "a").string());
  QLearningAgent loaded;
  loaded.load((dir / "a").string());
  EXPECT_TRUE(loaded.learnsBets());
  EXPECT_EQ(loaded.betSamples(8), TabularAgent::MIN_BET_SAMPLES);
  EXPECT_EQ(loaded.chooseBet(hot), 8);
  std::filesystem::remove_all(dir);
}

TEST(CountingTest, CountingAgentLearnsCountKeyedStatesAndAuditsAtZero) {
  TabularAgent::Hyperparameters p;
  p.useCount = true;
  p.learnBets = true;
  p.alphaSchedule = AlphaSchedule::VISIT;
  p.alphaOmega = 1.0;
  p.epsilonDecay = 0.99995;
  auto agent = std::make_shared<QLearningAgent>(p, 2u);

  TrainingConfig config;
  config.numEpisodes = 200'000;
  config.evalFrequency = 100'000;
  config.evalGames = 500;
  config.verbose = false;
  config.saveCheckpoints = false;
  config.logDir = "";
  config.seed = 4;
  config.betCalibrationRounds = 50'000;
  config.gameRules = GameRules::vegasStrip();
  config.gameRules.maxBetUnits = 8;
  Trainer trainer(agent, config);
  trainer.trainEpisodes(config.numEpisodes);

  State neutral(16, 10, false, false, true);
  neutral.countBucket = NEUTRAL_COUNT_BUCKET;
  State hot = neutral;
  hot.countBucket = 8;
  EXPECT_GT(agent->getVisitCount(neutral, Action::HIT), 0u);
  EXPECT_GT(agent->getVisitCount(hot, Action::HIT), 0u);
  EXPECT_EQ(agent->getVisitCount(State(16, 10, false, false, true), Action::HIT), 0u);

  // Audited at a true count of zero rather than at "untracked", where a
  // counting agent has no policy at all.
  PolicyAudit audit = Evaluator(config.gameRules).auditPolicy(agent.get());
  EXPECT_GT(audit.twoCard.scored, 100u);

  uint64_t calibrated = 0;
  for (int b = 1; b <= 10; ++b) calibrated += agent->betSamples(b);
  EXPECT_EQ(calibrated, config.betCalibrationRounds);
}
