#include "CountingAgent.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>

namespace blackjack {
namespace ai {

namespace {
constexpr size_t kBlindStates = 4096; ///< Hashes of count-blind states fit in 12 bits
} // namespace

CountingAgent::Prior CountingAgent::priorFromString(const std::string &s) {
  if (s == "none") return Prior::NONE;
  if (s == "frozen") return Prior::FROZEN;
  if (s == "pooled") return Prior::POOLED;
  throw std::invalid_argument("count_prior must be none, frozen or pooled, got '" + s + "'");
}

CountingAgent::Representation CountingAgent::representationFromString(const std::string &s) {
  if (s == "table") return Representation::TABLE;
  if (s == "linear") return Representation::LINEAR;
  throw std::invalid_argument("representation must be table or linear, got '" + s + "'");
}

CountingAgent::Exploration CountingAgent::explorationFromString(const std::string &s) {
  if (s == "epsilon") return Exploration::EPSILON;
  if (s == "ucb") return Exploration::UCB;
  if (s == "exploring-starts") return Exploration::EXPLORING_STARTS;
  throw std::invalid_argument(
      "exploration must be epsilon, ucb or exploring-starts, got '" + s + "'");
}

CountingAgent::Target CountingAgent::targetFromString(const std::string &s) {
  if (s == "td") return Target::TD;
  if (s == "monte-carlo") return Target::MONTE_CARLO;
  throw std::invalid_argument("target must be td or monte-carlo, got '" + s + "'");
}

CountingAgent::CountingAgent(const Hyperparameters &params, const Options &options,
                             std::optional<uint32_t> seed)
    : QLearningAgent(params, seed), options_(options), prior_(0.0) {
  if (!params.useCount) {
    throw std::invalid_argument("agent = counting needs use_count = true");
  }
  const bool sampleMeans =
      params.alphaSchedule == AlphaSchedule::VISIT && params.alphaOmega == 1.0;
  if (options_.prior != Prior::NONE && !sampleMeans) {
    // The shrinkage (k Q0 + n Q) / (k + n) is only a weighted mean of samples
    // when Q itself is one.
    throw std::invalid_argument("count_prior needs alpha_schedule = visit, alpha_omega = 1");
  }
  if (options_.prior != Prior::NONE && options_.representation == Representation::LINEAR) {
    throw std::invalid_argument("count_prior applies to the table representation only");
  }
  if (options_.priorVisits <= 0) {
    throw std::invalid_argument("count_prior_visits must be > 0");
  }
  if (options_.representation == Representation::LINEAR) fits_.resize(kBlindStates);
  if (options_.prior == Prior::FROZEN) {
    if (options_.priorModel.empty()) {
      throw std::invalid_argument("count_prior = frozen needs count_prior_model");
    }
    loadPrior(options_.priorModel);
  }
}

void CountingAgent::loadPrior(const std::string &modelPath) {
  prior_.loadFromBinary(modelPath + ".qtable");
}

State CountingAgent::blind(const State &s) {
  State b = s;
  b.countBucket = COUNT_UNTRACKED;
  return b;
}

double CountingAgent::trueCountOf(const State &s) {
  // The middle of the bucket's interval: floor(TC) = t means TC in [t, t+1).
  return s.countBucket == COUNT_UNTRACKED ? 0.0 : trueCountOfBucket(s.countBucket) + 0.5;
}

double CountingAgent::Fit::at(double x) const {
  if (n == 0) return 0.0; // the table's default, so untried reads alike
  const double meanY = sy / n;
  const double sxxc = sxx - sx * sx / n;
  if (n < 2 || sxxc < 1e-9) return meanY; // one count seen: no slope yet
  const double beta = (sxy - sx * sy / n) / sxxc;
  return meanY + beta * (x - sx / n);
}

double CountingAgent::value(const State &s, Action a) const {
  if (options_.representation == Representation::LINEAR) {
    return fits_[blind(s).hash()][static_cast<size_t>(a)].at(trueCountOf(s));
  }
  const double own = qTable_.get(s, a);
  if (options_.prior == Prior::NONE) return own;
  const State b = blind(s);
  if (prior_.getVisitCount(b, a) == 0) return own;
  const double n = qTable_.getVisitCount(s, a);
  const double k = options_.priorVisits;
  return (k * prior_.get(b, a) + n * own) / (k + n);
}

uint32_t CountingAgent::ownVisits(const State &s, Action a) const {
  if (options_.representation == Representation::LINEAR) {
    return static_cast<uint32_t>(fits_[blind(s).hash()][static_cast<size_t>(a)].n);
  }
  return qTable_.getVisitCount(s, a);
}

double CountingAgent::getQValue(const State &state, Action action) const {
  return value(state, action);
}

uint32_t CountingAgent::getVisitCount(const State &state, Action action) const {
  double n = ownVisits(state, action);
  if (options_.prior != Prior::NONE && prior_.getVisitCount(blind(state), action) > 0) {
    n += options_.priorVisits;
  }
  return n >= std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(n);
}

Action CountingAgent::greedy(const State &s, const std::vector<Action> &valid) const {
  // Same tie-break as PolicyTable::getMaxAction: the first of the best.
  double best = std::numeric_limits<double>::lowest();
  Action choice = valid[0];
  for (Action a : valid) {
    const double q = value(s, a);
    if (q > best) {
      best = q;
      choice = a;
    }
  }
  return choice;
}

double CountingAgent::maxValue(const State &s, const std::vector<Action> &valid) const {
  double best = std::numeric_limits<double>::lowest();
  for (Action a : valid) best = std::max(best, value(s, a));
  return best;
}

double CountingAgent::bootstrapValue(const State &nextState,
                                     const std::vector<Action> &validNextActions) const {
  return maxValue(nextState, validNextActions);
}

Action CountingAgent::chooseAction(const State &state, const std::vector<Action> &valid,
                                   bool training) {
  if (valid.empty()) throw std::invalid_argument("No valid actions provided");
  const bool first = firstDecision_;
  firstDecision_ = false;
  if (!training) return greedy(state, valid);

  switch (options_.exploration) {
  case Exploration::EXPLORING_STARTS:
    if (first) {
      std::uniform_int_distribution<size_t> pick(0, valid.size() - 1);
      return valid[pick(rng_)];
    }
    return greedy(state, valid);
  case Exploration::UCB: {
    double total = 0.0;
    for (Action a : valid) {
      const uint32_t n = ownVisits(state, a);
      if (n == 0) return a; // try everything once
      total += n;
    }
    double best = std::numeric_limits<double>::lowest();
    Action choice = valid[0];
    for (Action a : valid) {
      const double bonus = options_.ucbC * std::sqrt(std::log(total) / ownVisits(state, a));
      const double score = value(state, a) + bonus;
      if (score > best) {
        best = score;
        choice = a;
      }
    }
    return choice;
  }
  case Exploration::EPSILON:
    break;
  }
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  if (coin(rng_) < epsilon_) {
    std::uniform_int_distribution<size_t> pick(0, valid.size() - 1);
    return valid[pick(rng_)];
  }
  return greedy(state, valid);
}

void CountingAgent::update(const State &s, Action a, double target) {
  if (options_.representation == Representation::LINEAR) {
    Fit &f = fits_[blind(s).hash()][static_cast<size_t>(a)];
    const double x = trueCountOf(s);
    f.n += 1;
    f.sx += x;
    f.sxx += x * x;
    f.sy += target;
    f.sxy += x * target;
    return;
  }
  const double q = qTable_.get(s, a);
  qTable_.set(s, a, q + stepSize(qTable_.getVisitCount(s, a)) * (target - q));
  if (options_.prior == Prior::POOLED) {
    const State b = blind(s);
    const double p = prior_.get(b, a);
    prior_.set(b, a, p + stepSize(prior_.getVisitCount(b, a)) * (target - p));
  }
}

void CountingAgent::learn(const Experience &experience) {
  if (options_.target == Target::MONTE_CARLO) {
    learnMonteCarlo(experience);
    return;
  }
  double target = experience.reward;
  if (!experience.done) {
    double next = bootstrapValue(experience.nextState, nextActionsFor(experience));
    if (experience.secondNextState) {
      next += bootstrapValue(*experience.secondNextState, secondNextActionsFor(experience));
    }
    target += params_.discountFactor * next;
  }
  update(experience.state, experience.action, target);
  decayEpsilon();
  ++stepCount_;
  if (experience.done) firstDecision_ = true;
}

void CountingAgent::learnMonteCarlo(const Experience &experience) {
  // The Trainer hands over a round's decisions in order: one chain, or for a
  // split the SPLIT itself, then the first hand's chain, then the second's,
  // each chain's last decision terminal with that hand's reward.
  round_.push_back(experience);
  const bool split = round_.front().secondNextState.has_value();
  int done = 0;
  for (const Experience &e : round_) done += e.done ? 1 : 0;
  const bool complete = split ? (round_.front().done || done >= 2) : experience.done;
  if (!complete) return;

  const double g = params_.discountFactor;
  std::vector<double> ret(round_.size(), 0.0);
  // Return of each decision in a chain [from, to]: discounted reward at its end.
  auto chain = [&](size_t from, size_t to) {
    for (size_t k = to + 1; k-- > from;) {
      const bool last = k == to || round_[k].done;
      ret[k] = round_[k].reward + (last ? 0.0 : g * ret[k + 1]);
    }
  };
  if (!split) {
    chain(0, round_.size() - 1);
  } else if (round_.front().done) {
    ret[0] = round_[0].reward; // split aces settle on the spot
  } else {
    size_t endFirst = 1;
    while (endFirst < round_.size() && !round_[endFirst].done) ++endFirst;
    chain(1, endFirst);
    if (endFirst + 1 < round_.size()) chain(endFirst + 1, round_.size() - 1);
    const double second = endFirst + 1 < round_.size() ? ret[endFirst + 1] : 0.0;
    ret[0] = round_[0].reward + g * (ret[1] + second);
  }
  for (size_t k = 0; k < round_.size(); ++k) {
    update(round_[k].state, round_[k].action, ret[k]);
    decayEpsilon();
    ++stepCount_;
  }
  round_.clear();
  firstDecision_ = true;
}

void CountingAgent::save(const std::string &filepath) const {
  TabularAgent::save(filepath);
  if (options_.prior != Prior::NONE) prior_.saveToBinary(filepath + ".prior");
  if (options_.representation == Representation::LINEAR) {
    std::ofstream out(filepath + ".linear");
    if (!out) throw std::runtime_error("Cannot write " + filepath + ".linear");
    out << std::setprecision(17);
    for (size_t h = 0; h < fits_.size(); ++h) {
      for (size_t a = 0; a < PolicyTable::NUM_ACTIONS; ++a) {
        const Fit &f = fits_[h][a];
        if (f.n == 0) continue;
        out << h << ' ' << a << ' ' << f.n << ' ' << f.sx << ' ' << f.sxx << ' ' << f.sy
            << ' ' << f.sxy << '\n';
      }
    }
  }
}

void CountingAgent::load(const std::string &filepath) {
  TabularAgent::load(filepath);
  if (options_.prior != Prior::NONE) prior_.loadFromBinary(filepath + ".prior");
  if (options_.representation == Representation::LINEAR) {
    std::ifstream in(filepath + ".linear");
    if (!in) throw std::runtime_error("Cannot read " + filepath + ".linear");
    fits_.assign(kBlindStates, {});
    size_t h, a;
    Fit f;
    while (in >> h >> a >> f.n >> f.sx >> f.sxx >> f.sy >> f.sxy) {
      if (h < fits_.size() && a < PolicyTable::NUM_ACTIONS) fits_[h][a] = f;
    }
  }
}

void CountingAgent::reset() {
  TabularAgent::reset();
  if (options_.prior == Prior::POOLED) prior_.clear();
  if (options_.representation == Representation::LINEAR) fits_.assign(kBlindStates, {});
  round_.clear();
  firstDecision_ = true;
}

} // namespace ai
} // namespace blackjack
