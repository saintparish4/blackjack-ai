#pragma once

#include "../ai/Agent.hpp"
#include "../game/BlackjackGame.hpp"
#include "../game/GameRules.hpp"
#include <map>

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

  EvaluationResult()
      : gamesPlayed(0), handsPlayed(0), wins(0), losses(0), pushes(0),
        blackjacks(0), busts(0), winRate(0.0), lossRate(0.0), pushRate(0.0),
        avgReward(0.0), bustRate(0.0), strategyAccuracy(0.0) {}
};

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
  bool isCorrectAction(const ai::State &state, ai::Action action,
                       bool allowSurrender = true) const;

private:
  // Strategy tables: [player total][dealer up card] -> Action
  std::map<std::pair<int, int>, ai::Action> hardStrategy_;
  std::map<std::pair<int, int>, ai::Action> softStrategy_;

  void initializeHardStrategy();
  void initializeSoftStrategy();
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