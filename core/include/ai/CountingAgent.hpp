#pragma once

/**
 * @file CountingAgent.hpp
 * @brief The learners experiment 005 compares, as options on one agent.
 *
 * Experiment 002's learned counter put the true-count bucket in a Q-table
 * and learned each bucket from scratch. It failed to find the index plays.
 * The EoR analysis says why (experiments/005-index-plays/theory.txt): an
 * index decision moves the EV by under 1% per true count, so telling the
 * two plays apart at one bucket takes tens of thousands of samples *of the
 * less-played action*, and epsilon-greedy exploration supplies a few
 * percent of them. Each option here attacks one factor of that product:
 *
 *  - prior: shrink each bucket toward a count-blind estimate with weight
 *    `priorVisits`, so a bucket with little data plays like basic strategy
 *    instead of like noise. `frozen` reads the prior from a saved count-blind
 *    model; that is warm-starting with a pseudo-count, and also exactly
 *    "deviation learning" (Q0 fixed, only the per-bucket deviation trained),
 *    since both give Q = (k Q0 + sum of targets) / (k + n). `pooled` learns
 *    the prior online from every bucket at once.
 *  - representation: `linear` fits Q(s, a) = w0 + w1 x TC per state and
 *    action, pooling every bucket into two parameters. An index is where two
 *    such lines cross, which is the shape the EoR analysis predicts.
 *  - exploration: `ucb` explores in proportion to uncertainty; `exploring-
 *    starts` plays the first decision of every round uniformly at random and
 *    the rest greedily. Every Illustrious 18 decision is a first decision.
 *  - target: `monte-carlo` updates on the round's actual return instead of a
 *    bootstrapped estimate (with visit-count steps, Q is then the plain mean
 *    of returns: every-visit Monte Carlo control).
 *
 * Every option at its first value reproduces QLearningAgent with use_count.
 * Requires the visit-count step size with omega = 1, so a Q-value is a sample
 * mean and the shrinkage formula is exact.
 */

#include "QLearningAgent.hpp"
#include <array>
#include <string>
#include <vector>

namespace blackjack {
namespace ai {

class CountingAgent : public QLearningAgent {
public:
  enum class Prior { NONE, FROZEN, POOLED };
  enum class Representation { TABLE, LINEAR };
  enum class Exploration { EPSILON, UCB, EXPLORING_STARTS };
  enum class Target { TD, MONTE_CARLO };

  struct Options {
    Prior prior = Prior::NONE;
    double priorVisits = 100.0; ///< Pseudo-count k of the prior
    std::string priorModel;     ///< FROZEN: path of a count-blind model
    Representation representation = Representation::TABLE;
    Exploration exploration = Exploration::EPSILON;
    double ucbC = 1.0;
    Target target = Target::TD;
  };

  CountingAgent(const Hyperparameters &params, const Options &options,
                std::optional<uint32_t> seed = std::nullopt);

  Action chooseAction(const State &state, const std::vector<Action> &validActions,
                      bool training = true) override;
  void learn(const Experience &experience) override;
  double getQValue(const State &state, Action action) const override;
  /** Samples behind getQValue: the bucket's own, plus the prior's weight
   *  when it has one. Zero means the value is the table's default. */
  uint32_t getVisitCount(const State &state, Action action) const override;
  std::string getName() const override { return "Counting Q-learning"; }

  void save(const std::string &filepath) const override;
  void load(const std::string &filepath) override;
  void reset() override;

  /** FROZEN prior: load a count-blind model saved by any tabular agent. */
  void loadPrior(const std::string &modelPath);

  const Options &options() const { return options_; }

  static Prior priorFromString(const std::string &s);
  static Representation representationFromString(const std::string &s);
  static Exploration explorationFromString(const std::string &s);
  static Target targetFromString(const std::string &s);

protected:
  double bootstrapValue(const State &nextState,
                        const std::vector<Action> &validNextActions) const override;

private:
  /// Least-squares sufficient statistics for one (state, action) of the
  /// linear representation, in x = the bucket's true count.
  struct Fit {
    double n = 0, sx = 0, sxx = 0, sy = 0, sxy = 0;
    double at(double x) const;
  };

  Options options_;
  PolicyTable prior_; ///< FROZEN or POOLED estimates, keyed by count-blind state
  std::vector<std::array<Fit, PolicyTable::NUM_ACTIONS>> fits_; ///< LINEAR
  bool firstDecision_ = true;
  std::vector<Experience> round_; ///< MONTE_CARLO: the round so far

  static State blind(const State &s);
  static double trueCountOf(const State &s);
  double value(const State &s, Action a) const;
  uint32_t ownVisits(const State &s, Action a) const;
  Action greedy(const State &s, const std::vector<Action> &valid) const;
  double maxValue(const State &s, const std::vector<Action> &valid) const;
  void update(const State &s, Action a, double target);
  void learnMonteCarlo(const Experience &experience);
};

} // namespace ai
} // namespace blackjack
