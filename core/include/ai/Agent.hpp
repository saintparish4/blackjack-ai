#pragma once

#include "State.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace blackjack {
namespace ai {

enum class Action : uint8_t { HIT = 0, STAND = 1, DOUBLE = 2, SPLIT = 3, SURRENDER = 4 };

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
  case Action::SURRENDER:
    return "SURRENDER";
  default:
    return "UNKNOWN";
  }
}

/** One step: (state, action, reward, next_state, done, valid_next_actions). */
struct Experience {
  State state;
  Action action;
  double reward;
  State nextState;
  bool done;
  std::vector<Action> validNextActions;

  /// A split moves into two states at once: the first hand (nextState) and
  /// the second. The split is worth both, so the target bootstraps from the
  /// sum. Unset for every other action.
  std::optional<State> secondNextState;
  std::vector<Action> secondValidNextActions;

  Experience(const State &s, Action a, double r, const State &ns, bool d,
             std::vector<Action> vna = {})
      : state(s), action(a), reward(r), nextState(ns), done(d),
        validNextActions(std::move(vna)) {}
};

/** Base interface for learning agents. */
class Agent {
public:
  virtual ~Agent() = default;

  /** training: true = may explore, false = exploit only. */
  virtual Action chooseAction(const State &state,
                              const std::vector<Action> &validActions,
                              bool training = true) = 0;

  virtual void learn(const Experience &experience) = 0;
  virtual double getQValue(const State &state, Action action) const = 0;
  virtual void save(const std::string &filepath) const = 0;
  virtual void load(const std::string &filepath) = 0;
  virtual std::string getName() const = 0;

  virtual double getExplorationRate() const { return 0.0; }
  virtual size_t getStateCount() const { return 0; }

  /** How many learning updates this (state, action) pair has received.
   *
   *  Lets a policy audit tell "the agent evaluated this action and disliked it"
   *  apart from "this action was never legal here, so its Q is still the
   *  initial default". Defaults to 1 so fixed-policy agents, which have nothing
   *  to learn, are never reported as untrained. */
  virtual uint32_t getVisitCount(const State &, Action) const { return 1; }
};
} // namespace ai
} // namespace blackjack