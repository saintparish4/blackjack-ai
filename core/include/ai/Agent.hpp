#pragma once

#include "State.hpp"
#include <string>
#include <vector>

namespace blackjack {
namespace ai {
/** @brief Available actions in blackjack */
enum class Action : uint8_t { HIT = 0, STAND = 1, DOUBLE = 2, SPLIT = 3 };

/** @brief Convert action to string */
inline std::string actionToString(Action action) {
  switch (action) {
  case Action::HIT:
    return "HIT";
  case Action::STAND:
    return "STAND";
  case Action::DOUBLE:
    return "DOUBLE";
  case Action::SPLIT:
    return "SPLIT";
  default:
    return "UNKNOWN";
  }
}

/** @brief Experience tuple for learning
 *
 * Represents a single step in the environment: (state, action, reward,
 * next_state, done)
 */
struct Experience {
  State state;
  Action action;
  double reward;
  State nextState;
  bool done;

  Experience(const State &s, Action a, double r, const State &ns, bool d)
      : state(s), action(a), reward(r), nextState(ns), done(d) {}
};

/** @brief Abstract base class for all agents
 *
 * Defines the interface that all learning agents must implement
 */
class Agent {
public:
  virtual ~Agent() = default;

  /** @brief Choose an action given current state
   *
   * @param state Current game state
   * @param validAction List of legal actions in this state
   * @param training If true, may explore; if false, exploit only
   * @return Selected action
   */
  virtual Action chooseAction(const State &state,
                              const std::vector<Action> &validActions,
                              bool training = true) = 0;

  /** @brief Learn from experience
   *
   * @param experience The (s, a, r, s', done) tuple
   */
  virtual void learn(const Experience &experience) = 0;

  /**
   * @brief Get the Q-value for a state-action pair
   *
   * @param state The state
   * @param action The action
   * @return Q-value estimate
   */
  virtual double getQValue(const State &state, Action action) const = 0;

  /**
   * @brief Save agent's learned policy to file
   */
  virtual void save(const std::string &filepath) const = 0;

  /**
   * @brief Load agent's policy from file
   */
  virtual void load(const std::string &filepath) = 0;

  /**
   * @brief Get agent's name/type
   */
  virtual std::string getName() const = 0;
};
} // namespace ai
} // namespace blackjack