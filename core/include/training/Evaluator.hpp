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
  size_t gamesPlayed;
  size_t wins;
  size_t losses;
  size_t pushes;
  size_t blackjacks;
  size_t busts;

  double winRate;   ///< Wins / total games
  double lossRate;  ///< Losses / total games
  double pushRate;  ///< Pushes / total games
  double avgReward; ///< Average reward per game
  double bustRate;  ///< Bust / total games

  double strategyAccuracy; ///< Match with basic strategy (0-1)

  EvaluationResult()
      : gamesPlayed(0), wins(0), losses(0), pushes(0), blackjacks(0), busts(0),
        winRate(0.0), lossRate(0.0), pushRate(0.0), avgReward(0.0),
        bustRate(0.0), strategyAccuracy(0.0) {}
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
  ai::Action getAction(const ai::State &state) const;

  /**
   * @brief Check if action matches basic strategy
   */
  bool isCorrectAction(const ai::State &state, ai::Action action) const;

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