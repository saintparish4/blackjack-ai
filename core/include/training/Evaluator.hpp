#pragma once

#include "../ai/Agent.hpp"
#include "../game/BlackjackGame.hpp"
#include "../game/GameRules.hpp"
#include <array>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <vector>

namespace blackjack {
namespace training {
/**
 * @brief Result of agent evaluation
 */
struct EvaluationResult {
  size_t gamesPlayed; ///< Rounds dealt
  size_t handsPlayed; ///< Settled hands; exceeds gamesPlayed when hands split
  size_t wins;
  size_t losses;
  size_t pushes;
  size_t blackjacks;
  size_t busts;

  // Rates are per settled hand, not per round: a split round settles two hands
  // independently, so dividing by gamesPlayed would let the three rates sum
  // above 1.0.
  double winRate;   ///< Wins / total hands
  double lossRate;  ///< Losses / total hands
  double pushRate;  ///< Pushes / total hands
  double avgReward; ///< Average reward per round (per initial bet)
  double bustRate;  ///< Busts / total hands

  double strategyAccuracy; ///< Match with basic strategy (0-1)

  /// Standard error of avgReward. Blackjack EV differences are small relative
  /// to per-round variance (~1.15), so a bare mean is not comparable between
  /// two agents without it.
  double rewardStdErr;

  // ---- Betting. With flat bets these equal avgReward / rewardStdErr. ----

  double avgBet = 1.0;          ///< Mean units wagered per round
  /// Units won per unit wagered: the player's edge including bet sizing. A
  /// counter's avgReward ignores its bets and measures only how it plays.
  double evPerUnitBet = 0.0;
  double evPerUnitStdErr = 0.0; ///< Delta-method standard error of the ratio
  double wonPerRound = 0.0;     ///< Units won per round (bet x reward)
  /// Standard deviation of units won per round. Bigger bets raise it along
  /// with wonPerRound; desirabilityIndex() weighs one against the other.
  double wonStdDev = 0.0;

  /** Schlesinger's desirability index, 1000 x mean / SD of units won per
   *  round: a risk-adjusted score that doesn't care how big the unit is. */
  double desirabilityIndex() const {
    return wonStdDev > 0.0 ? 1000.0 * wonPerRound / wonStdDev : 0.0;
  }

  /// Rounds and summed per-unit reward by the true-count bucket the round was
  /// dealt at: the policy's edge as a function of the count.
  std::array<uint64_t, 16> roundsByCount{};
  std::array<double, 16> rewardByCount{};

  EvaluationResult()
      : gamesPlayed(0), handsPlayed(0), wins(0), losses(0), pushes(0),
        blackjacks(0), busts(0), winRate(0.0), lossRate(0.0), pushRate(0.0),
        avgReward(0.0), bustRate(0.0), strategyAccuracy(0.0),
        rewardStdErr(0.0) {}
};

/**
 * @brief An agent's EV next to the two baselines that give it meaning.
 *
 * All three are measured by the same Evaluator on the same rules, so they are
 * directly comparable; see Evaluator::compareAgainstBaselines.
 */
struct BaselineComparison {
  EvaluationResult agent;
  EvaluationResult book;   ///< BasicStrategyAgent
  EvaluationResult random; ///< RandomAgent

  /// Share of the random-to-book EV range the agent recovers: 0 plays like
  /// random, 1 plays like the book. NaN when the range is degenerate.
  double gapRecovered = 0.0;
};

/**
 * @brief Which hand shape a policy audit is scoring.
 *
 * The two are genuinely different policies stored in different Q-table slots,
 * because State::hash() packs canDouble. Auditing only one of them measures
 * half the agent.
 */
enum class HandContext {
  TwoCard,   ///< Opening hand: DOUBLE and SURRENDER are legal.
  MultiCard, ///< Already hit at least once: only HIT and STAND remain.
  Pair,      ///< Opening pair: SPLIT is legal too. Scored against the pair chart.
};

/** One state a policy audit scores, paired with the actions legal there. */
struct AuditState {
  ai::State state;
  std::vector<ai::Action> validActions;
};

/** Audit totals for a single hand context. */
struct ContextAudit {
  size_t scored = 0;    ///< States where every legal action had been trained
  size_t matches = 0;   ///< Of those, how many matched basic strategy
  size_t untrained = 0; ///< Skipped: some legal action never received an update

  double accuracy() const {
    return scored > 0 ? static_cast<double>(matches) / scored : 0.0;
  }
};

/**
 * @brief Policy audit split by hand context, plus a combined accuracy.
 *
 * Reporting the two contexts separately is the point: a single blended number
 * hides that the opening-hand policy (where doubling lives) and the
 * already-hit policy are learned independently and converge at different rates.
 */
struct PolicyAudit {
  ContextAudit twoCard;
  ContextAudit multiCard;
  /// Pairs were left out of the audit until an agent that over-split 27 of
  /// 100 pair cells still scored well: the leak was invisible to it.
  ContextAudit pairs;
  double accuracy = 0.0; ///< Over all scored states in every context
};

/** All three contexts, in reporting order. */
constexpr HandContext kAllHandContexts[] = {HandContext::TwoCard,
                                            HandContext::MultiCard,
                                            HandContext::Pair};

/**
 * @brief Actions the rules permit in this state.
 *
 * Derived from state.canDouble / state.canSplit so there is exactly one
 * definition of "legal here". Three separate copies of this logic previously
 * drifted apart, which is how the audit came to offer DOUBLE only on totals
 * 9-11 while the game engine allowed it on any two-card hand.
 */
std::vector<ai::Action> legalActionsFor(const ai::State &state,
                                        const GameRules &rules);

/**
 * @brief Enumerate the states a policy audit should score for a hand context.
 *
 * Offers every action the rules actually permit, rather than a hand-picked
 * subset. Restricting DOUBLE to totals 9-11 (as this code used to) means the
 * agent is never scored on the error of doubling a 16.
 */
std::vector<AuditState> buildAuditStates(const GameRules &rules,
                                         HandContext context);

/**
 * @brief Basic strategy lookup table
 *
 * Multi-deck, total-dependent basic strategy for this engine's rules (dealer
 * peeks, one split, split aces one card, no double after split), with the
 * S17/H17 differences taken from the rules. Single- and double-deck presets
 * are graded against the same multi-deck chart; the exact per-rule answer
 * needs a solver (see the roadmap).
 */
class BasicStrategy {
public:
  explicit BasicStrategy(const GameRules &rules = GameRules{});

  /**
   * @brief Get optimal action for state
   */
  /**
   * @param allowSurrender  When false, states whose book play is SURRENDER
   *   fall back to their no-surrender play (HIT). Pass the value of
   *   GameRules::surrender: scoring an agent against a SURRENDER
   *   recommendation it was never allowed to learn measures nothing.
   */
  ai::Action getAction(const ai::State &state,
                       bool allowSurrender = true) const;

  /**
   * @brief Check if action matches basic strategy
   */
  /**
   * @param allowSurrender  As getAction.
   *
   * A DOUBLE recommendation is satisfied by HIT only when the state cannot
   * actually double (state.canDouble == false), which is the real situation on
   * a hand that has already hit. Forgiving it unconditionally credits the agent
   * for missing the highest-EV play in the game.
   */
  bool isCorrectAction(const ai::State &state, ai::Action action,
                       bool allowSurrender = true) const;

  /**
   * @brief The play when the book says DOUBLE but the hand can't double.
   *
   * STAND on the "double, else stand" soft cells (soft 18 against a weak
   * card), HIT everywhere else. Treating every DOUBLE as HIT would call
   * hitting a three-card soft 18 against a 5 correct.
   */
  ai::Action fallbackWhenCannotDouble(const ai::State &state) const;

private:
  // Strategy tables: [player total][dealer up card] -> Action
  std::map<std::pair<int, int>, ai::Action> hardStrategy_;
  std::map<std::pair<int, int>, ai::Action> softStrategy_;
  // Soft cells where DOUBLE falls back to STAND rather than HIT.
  std::set<std::pair<int, int>> doubleElseStand_;
  // Hard cells where late surrender beats the table's play, when offered.
  std::set<std::pair<int, int>> surrender_;
  // [pair rank (A=1, T=10)][dealer up card] -> SPLIT, or the play when not
  // splitting. Without this the EV baseline below misplays every pair and
  // stops being a fair reference.
  std::map<std::pair<int, int>, ai::Action> pairStrategy_;

  void initializeHardStrategy(bool h17);
  void initializeSoftStrategy(bool h17);
  void initializePairStrategy();
};

/**
 * @brief Plays textbook basic strategy. Reference agent for EV comparison.
 *
 * Implements ai::Agent so it runs through the exact same Evaluator loop, game
 * engine and rules as a learned agent. A baseline computed any other way is
 * not comparable.
 */
class BasicStrategyAgent : public ai::Agent {
public:
  /** Plays the chart for these rules, surrendering only if they offer it. */
  explicit BasicStrategyAgent(const GameRules &rules = GameRules{})
      : strategy_(rules), allowSurrender_(rules.surrender) {}

  ai::Action chooseAction(const ai::State &state,
                          const std::vector<ai::Action> &validActions,
                          bool training = true) override;

  void learn(const ai::Experience &) override {}
  double getQValue(const ai::State &, ai::Action) const override { return 0.0; }
  void save(const std::string &) const override {}
  void load(const std::string &) override {}
  std::string getName() const override { return "BasicStrategy"; }

private:
  BasicStrategy strategy_;
  bool allowSurrender_;
};

/**
 * One of the Illustrious 18: at a true count of `index` or above play
 * `atOrAbove`, below it play `below`. `pair` marks the two T,T split plays;
 * every other entry is a hard total (two cards or more).
 */
struct IndexPlay {
  int total;
  int dealer; ///< Ace = 1
  int index;
  ai::Action atOrAbove;
  ai::Action below;
  bool pair = false;
};

/**
 * Schlesinger's S17 multi-deck Illustrious 18 without insurance (the table
 * doesn't offer it): 17 plays, most valuable first. The one source for both
 * HiLoAgent's index plays and the targets a learned counter is scored on.
 */
const std::vector<IndexPlay> &illustrious18();

/**
 * @brief Textbook Hi-Lo counter: basic strategy plus a bet ramp, optionally
 * with the Illustrious 18 index plays.
 *
 * The reference a learned counter has to beat, run through the same Evaluator,
 * engine and rules. Insurance, the first of the Illustrious 18, is left out
 * because this table doesn't offer it; the other 17 are Schlesinger's S17
 * multi-deck indices.
 */
class HiLoAgent : public ai::Agent {
public:
  explicit HiLoAgent(const GameRules &rules = GameRules{}, bool indexPlays = false)
      : book_(rules), indexPlays_(indexPlays) {}

  ai::Action chooseAction(const ai::State &state,
                          const std::vector<ai::Action> &validActions,
                          bool training = true) override;
  int chooseBet(const ai::BetContext &ctx, bool training = false) override;
  bool usesCount() const override { return true; }

  void learn(const ai::Experience &) override {}
  double getQValue(const ai::State &, ai::Action) const override { return 0.0; }
  void save(const std::string &) const override {}
  void load(const std::string &) override {}
  std::string getName() const override {
    return indexPlays_ ? "Hi-Lo + Illustrious 18" : "Hi-Lo";
  }

  /** A common 1-8 ramp: 1 unit up to TC +1, then 2, 4, 6 and 8 from +2 to
   *  +5, capped at maxBet. */
  static int rampUnits(int trueCountFloor, int maxBet);

  /** The index play for this hand at this count, if one applies. */
  static std::optional<ai::Action> indexPlay(const ai::State &state,
                                             int trueCountFloor);

private:
  BasicStrategyAgent book_;
  bool indexPlays_;
};

/**
 * @brief Picks uniformly at random from the legal actions. Lower-bound baseline.
 *
 * Anchors the scale: without it, "-0.058 per round" has no reference telling
 * you whether it is near book play or near coin-flipping.
 */
class RandomAgent : public ai::Agent {
public:
  explicit RandomAgent(uint32_t seed = 42) : rng_(seed) {}

  ai::Action chooseAction(const ai::State &state,
                          const std::vector<ai::Action> &validActions,
                          bool training = true) override;

  void learn(const ai::Experience &) override {}
  double getQValue(const ai::State &, ai::Action) const override { return 0.0; }
  void save(const std::string &) const override {}
  void load(const std::string &) override {}
  std::string getName() const override { return "Random"; }

private:
  std::mt19937 rng_;
};

/**
 * @brief Evaluates agent performance
 *
 * Provides various metrics:
 * - Win/loss/push rates
 * - Average reward
 * - Strategy accuracy (vs basic strategy)
 * - Performance breakdown by state
 */
class Evaluator {
public:
  /**
   * @brief Construct evaluator with game rules
   *
   * @param seed Fixes the shoes every evaluate() call deals from. Each call
   *   still gets a fresh shoe (derived from the seed and a call counter), so a
   *   seeded run is reproducible without every evaluation replaying one deal.
   */
  explicit Evaluator(const GameRules &rules = GameRules{},
                     std::optional<uint32_t> seed = std::nullopt);

  /**
   * @brief Evaluate agent over multiple games
   *
   * @param agent Agent to evaluate
   * @param numGames Number of games to play
   * @param compareStrategy Compare with basic strategy
   * @return Evaluation metrics
   */
  EvaluationResult evaluate(ai::Agent *agent, size_t numGames,
                            bool compareStrategy = true);

  /**
   * @brief Evaluate an agent alongside basic strategy and random play.
   *
   * @param rounds Rounds per policy. Per-round reward has a standard deviation
   *   near 1.15, so anything much under 100,000 cannot resolve the gap between
   *   a good agent and the book.
   * @param baselineSeed Seeds RandomAgent's action choices.
   */
  BaselineComparison compareAgainstBaselines(ai::Agent *agent, size_t rounds,
                                             uint32_t baselineSeed = 1234);

  /**
   * @brief Teach a bet-learning agent its edge at each count.
   *
   * Plays `rounds` rounds greedily (no exploration, no Q updates) and feeds
   * each round's per-unit result to agent->learnBet at the count it was dealt
   * at. Measuring the greedy policy, not the exploring one used in training,
   * matters: ε = 0.05 of random moves costs a few percent per round and would
   * push every edge below zero.
   */
  void calibrateBets(ai::Agent *agent, size_t rounds);

  /**
   * @brief Compare agent's strategy with basic strategy via exhaustive iteration
   *
   * Iterates all valid (playerTotal 4-21) x (dealerCard 1-10) x (soft/hard)
   * states rather than random sampling, giving a deterministic accuracy score.
   *
   * @return Accuracy (0-1, fraction of states where agent matches basic strategy)
   */
  double compareWithBasicStrategy(ai::Agent *agent);

  /**
   * @brief Full policy audit, split by hand context and reporting coverage.
   *
   * Scores only states where every legal action has actually been trained; the
   * rest are counted as coverage gaps rather than silently graded off an
   * untouched Q-value.
   */
  PolicyAudit auditPolicy(ai::Agent *agent);

  /**
   * @brief Get basic strategy reference
   */
  const BasicStrategy &getBasicStrategy() const { return basicStrategy_; }

private:
  GameRules rules_;
  BasicStrategy basicStrategy_;
  std::optional<uint32_t> seed_;
  uint32_t evaluationCount_ = 0;

  /** Shoe seed for the next evaluate() call; nullopt when unseeded. */
  std::optional<uint32_t> nextDealSeed();

  /**
   * @brief Play one evaluation game.
   * @return One outcome per player hand (multiple after split).
   */
  std::vector<Outcome> playGame(ai::Agent *agent, BlackjackGame &game);

  /** Reshuffle if due, then read the count the next round's bet sees. */
  ai::BetContext betContext(BlackjackGame &game) const;
};

/**
 * @brief The state an audit should ask this agent about.
 *
 * A counting agent keys its Q-table by count, so it has no policy at all at
 * COUNT_UNTRACKED; audit it at a true count of zero, where basic strategy is
 * the right answer.
 */
inline ai::State auditView(const ai::Agent &agent, ai::State state) {
  if (agent.usesCount()) state.countBucket = ai::NEUTRAL_COUNT_BUCKET;
  return state;
}
} // namespace training
} // namespace blackjack