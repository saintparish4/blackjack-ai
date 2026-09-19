#include "TabularAgent.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace blackjack {
namespace ai {

AlphaSchedule alphaScheduleFromString(const std::string &name) {
  if (name == "constant") return AlphaSchedule::CONSTANT;
  if (name == "visit") return AlphaSchedule::VISIT;
  throw std::invalid_argument("alpha_schedule must be 'constant' or 'visit', got '" +
                              name + "'");
}

std::string toString(AlphaSchedule schedule) {
  return schedule == AlphaSchedule::VISIT ? "visit" : "constant";
}

TabularAgent::TabularAgent(const Hyperparameters &params,
                           std::optional<uint32_t> seed)
    : params_(params), qTable_(0.0), epsilon_(params.epsilon),
      rng_(seed ? *seed : std::random_device{}()), stepCount_(0) {
  if (!params_.isValid()) {
    throw std::invalid_argument("Invalid hyperparameters");
  }
}

Action TabularAgent::chooseAction(const State &state,
                                  const std::vector<Action> &validActions,
                                  bool training) {
  if (validActions.empty()) {
    throw std::invalid_argument("No valid actions provided");
  }

  if (training) {
    return epsilonGreedy(state, validActions);
  }
  return greedyAction(state, validActions);
}

const std::vector<Action> &
TabularAgent::nextActionsFor(const Experience &experience) {
  // Terminal or unknown next state: fall back to the two base actions
  static const std::vector<Action> fallback = {Action::HIT, Action::STAND};
  return experience.validNextActions.empty() ? fallback
                                             : experience.validNextActions;
}

const std::vector<Action> &
TabularAgent::secondNextActionsFor(const Experience &experience) {
  static const std::vector<Action> fallback = {Action::HIT, Action::STAND};
  return experience.secondValidNextActions.empty()
             ? fallback
             : experience.secondValidNextActions;
}

double TabularAgent::stepSize(uint32_t priorUpdates) const {
  if (params_.alphaSchedule == AlphaSchedule::CONSTANT) {
    return params_.learningRate;
  }
  const double n = static_cast<double>(priorUpdates) + 1.0;
  // ω = 1 is the common case and a plain division; std::pow on every update
  // cost about a quarter of training throughput.
  const double alpha =
      params_.alphaOmega == 1.0 ? 1.0 / n : std::pow(n, -params_.alphaOmega);
  return std::max(params_.alphaMin, alpha);
}

void TabularAgent::learn(const Experience &experience) {
  const State &state = experience.state;
  const Action action = experience.action;

  double currentQ = qTable_.get(state, action);
  double targetQ = experience.reward;
  if (!experience.done) {
    double next = bootstrapValue(experience.nextState, nextActionsFor(experience));
    if (experience.secondNextState) {
      next += bootstrapValue(*experience.secondNextState,
                             secondNextActionsFor(experience));
    }
    targetQ += params_.discountFactor * next;
  }

  // Q(s,a) ← Q + α[target - Q]
  const double alpha = stepSize(qTable_.getVisitCount(state, action));
  qTable_.set(state, action, currentQ + alpha * (targetQ - currentQ));

  decayEpsilon();
  stepCount_++;
}

int TabularAgent::chooseBet(const BetContext &ctx, bool) {
  if (!params_.learnBets) {
    return ctx.minBet;
  }
  const int bucket = ctx.countBucket != COUNT_UNTRACKED
                         ? ctx.countBucket
                         : countBucketFor(ctx.trueCount);
  const auto b = static_cast<size_t>(bucket);
  if (betSamples_[b] < MIN_BET_SAMPLES || betValue_[b] <= 0.0) {
    return ctx.minBet;
  }
  if (!params_.betProportional) {
    return ctx.maxBet;
  }
  const long units = std::lround(betValue_[b] / params_.betUnitEdge);
  return static_cast<int>(std::clamp<long>(units, ctx.minBet, ctx.maxBet));
}

void TabularAgent::learnBet(const BetContext &ctx, double perUnitReward) {
  const int bucket = ctx.countBucket != COUNT_UNTRACKED
                         ? ctx.countBucket
                         : countBucketFor(ctx.trueCount);
  const auto b = static_cast<size_t>(bucket);
  ++betSamples_[b];
  betValue_[b] += (perUnitReward - betValue_[b]) / static_cast<double>(betSamples_[b]);
}

double TabularAgent::getQValue(const State &state, Action action) const {
  return qTable_.get(state, action);
}

void TabularAgent::save(const std::string &filepath) const {
  std::string qtablePath = filepath + ".qtable";
  std::string metaPath = filepath + ".meta";

  qTable_.saveToBinary(qtablePath);

  std::ofstream metaFile(metaPath);
  if (!metaFile) {
    throw std::runtime_error("Cannot open meta file for writing");
  }

  // Default stream precision is 6 significant digits, which writes an
  // epsilon_decay of 0.9999985 as 0.999999: a decay rate that reaches
  // epsilon_min at a completely different point in training.
  metaFile << std::setprecision(10);
  metaFile << "agent_type: " << getName() << "\n";
  metaFile << "learning_rate: " << params_.learningRate << "\n";
  metaFile << "alpha_schedule: " << toString(params_.alphaSchedule) << "\n";
  metaFile << "alpha_omega: " << params_.alphaOmega << "\n";
  metaFile << "alpha_min: " << params_.alphaMin << "\n";
  metaFile << "discount_factor: " << params_.discountFactor << "\n";
  metaFile << "epsilon: " << epsilon_ << "\n";
  metaFile << "epsilon_min: " << params_.epsilonMin << "\n";
  metaFile << "epsilon_decay: " << params_.epsilonDecay << "\n";
  metaFile << "step_count: " << stepCount_ << "\n";
  metaFile << "state_space_size: " << qTable_.size() << "\n";
  metaFile << "use_count: " << (params_.useCount ? 1 : 0) << "\n";
  metaFile << "learn_bets: " << (params_.learnBets ? 1 : 0) << "\n";
  metaFile << "bet_proportional: " << (params_.betProportional ? 1 : 0) << "\n";
  metaFile << "bet_unit_edge: " << params_.betUnitEdge << "\n";
  for (int b = 1; b <= countBucketFor(MAX_TRUE_COUNT); ++b) {
    const auto i = static_cast<size_t>(b);
    if (betSamples_[i] == 0) continue;
    metaFile << "bet_value_" << b << ": " << betValue_[i] << "\n";
    metaFile << "bet_samples_" << b << ": " << betSamples_[i] << "\n";
  }

  metaFile.close();
}

void TabularAgent::load(const std::string &filepath) {
  std::string qtablePath = filepath + ".qtable";
  std::string metaPath = filepath + ".meta";

  qTable_.loadFromBinary(qtablePath);

  std::ifstream metaFile(metaPath);
  if (!metaFile) {
    throw std::runtime_error("Cannot open meta file for reading");
  }

  std::string line;
  while (std::getline(metaFile, line)) {
    size_t colonPos = line.find(':');
    if (colonPos == std::string::npos)
      continue;

    std::string key = line.substr(0, colonPos);
    std::string value = line.substr(colonPos + 2);

    if (key == "epsilon") {
      epsilon_ = std::stod(value);
    } else if (key == "step_count") {
      stepCount_ = std::stoull(value);
    } else if (key == "use_count") {
      // A counting model's Q-values are keyed by count; played count-blind it
      // would find every state untrained.
      params_.useCount = value == "1";
    } else if (key == "learn_bets") {
      params_.learnBets = value == "1";
    } else if (key == "bet_proportional") {
      params_.betProportional = value == "1";
    } else if (key == "bet_unit_edge") {
      params_.betUnitEdge = std::stod(value);
    } else if (key.rfind("bet_value_", 0) == 0 || key.rfind("bet_samples_", 0) == 0) {
      const bool isValue = key[4] == 'v';
      const size_t b = std::stoul(key.substr(isValue ? 10 : 12));
      if (b < betValue_.size()) {
        if (isValue) betValue_[b] = std::stod(value);
        else betSamples_[b] = std::stoull(value);
      }
    }
  }

  metaFile.close();
}

void TabularAgent::reset() {
  qTable_.clear();
  epsilon_ = params_.epsilon;
  stepCount_ = 0;
  betValue_.fill(0.0);
  betSamples_.fill(0);
}

Action TabularAgent::epsilonGreedy(const State &state,
                                   const std::vector<Action> &validActions) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double rand = dist(rng_);

  if (rand < epsilon_) {
    std::uniform_int_distribution<size_t> actionDist(0,
                                                     validActions.size() - 1);
    return validActions[actionDist(rng_)];
  }
  return greedyAction(state, validActions);
}

Action TabularAgent::greedyAction(const State &state,
                                  const std::vector<Action> &validActions) const {
  return qTable_.getMaxAction(state, validActions);
}

void TabularAgent::decayEpsilon() {
  epsilon_ *= params_.epsilonDecay;
  epsilon_ = std::max(epsilon_, params_.epsilonMin);
}

} // namespace ai
} // namespace blackjack
