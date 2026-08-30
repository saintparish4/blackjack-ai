#pragma once

#include "../ai/Agent.hpp"
#include "../game/BlackjackGame.hpp"
#include "../game/GameRules.hpp"
#include <map>
#include <random>
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

  EvaluationResult()
      : gamesPlayed(0), handsPlayed(0), wins(0), losses(0), pushes(0),
        blackjacks(0), busts(0), winRate(0.0), lossRate(0.0), pushRate(0.0),
        avgReward(0.0), bustRate(0.0), strategyAccuracy(0.0),
        rewardStdErr(0.0) {}
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
  double accuracy = 0.0; ///< Over all scored states in both contexts
};

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
 * Mathematically optimal decisions for blackjack.
 * Used to compare learned strategy with optimal strategy.
 */
class BasicStrategy {
public:
  BasicStrategy();

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

private:
  // Strategy tables: [player total][dealer up card] -> Action
  std::map<std::pair<int, int>, ai::Action> hardStrategy_;
  std::map<std::pair<int, int>, ai::Action> softStrategy_;
  // [pair rank (A=1, T=10)][dealer up card] -> SPLIT, or the play when not
  // splitting. Without this the EV baseline below misplays every pair and
  // stops being a fair reference.
  std::map<std::pair<int, int>, ai::Action> pairStrategy_;

  void initializeHardStrategy();
  void initializeSoftStrategy();
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
  explicit BasicStrategyAgent(bool allowSurrender = false)
      : allowSurrender_(allowSurrender) {}

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
   */
  explicit Evaluator(const GameRules &rules = GameRules{});

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

  /**
   * @brief Play one evaluation game.
   * @return One outcome per player hand (multiple after split).
   */
  std::vector<Outcome> playGame(ai::Agent *agent, BlackjackGame &game);
};
} // namespace training
} // namespace blackjack