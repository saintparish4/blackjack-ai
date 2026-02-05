#include "QLearningAgent.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace blackjack {
namespace ai {
QLearningAgent::QLearningAgent(const Hyperparameters &params)
    : params_(params), qTable_(0.0), epsilon_(params.epsilon),
      rng_(std::random_device{}()), stepCount_(0) {
  if (!params_.isValid()) {
    throw std::invalid_argument("Invalid hyperparameters");
  }
}

Action QLearningAgent::chooseAction(const State &state,
                                    const std::vector<Action> &validActions,
                                    bool training) {
  if (validActions.empty()) {
    throw std::invalid_argument("No valid actions provided");
  }

  if (training) {
    return epsilonGreedy(state, validActions);
  } else {
    return greedyAction(state, validActions);
  }
}

void QLearningAgent::learn(const Experience &experience) {
  const State &state = experience.state;
  const Action action = experience.action;
  const double reward = experience.reward;
  const State &nextState = experience.nextState;
  const bool done = experience.done;

  double currentQ = qTable_.get(state, action);
  double targetQ;

  if (done) {
    targetQ = reward;
  } else {
    std::vector<Action> allActions = {Action::HIT, Action::STAND,
                                      Action::DOUBLE, Action::SPLIT};
    double maxNextQ = qTable_.getMaxQ(nextState, allActions);
    targetQ = reward + params_.discountFactor * maxNextQ;
  }

  // Q(s,a) ← Q + α[target - Q]
  double newQ = currentQ + params_.learningRate * (targetQ - currentQ);
  qTable_.set(state, action, newQ);

  decayEpsilon();
  stepCount_++;
}

double QLearningAgent::getQValue(const State &state, Action action) const {
  return qTable_.get(state, action);
}

void QLearningAgent::save(const std::string &filepath) const {
  std::string qtablePath = filepath + ".qtable";
  std::string metaPath = filepath + ".meta";

  qTable_.saveToBinary(qtablePath);

  std::ofstream metaFile(metaPath);
  if (!metaFile) {
    throw std::runtime_error("Cannot open meta file for writing");
  }

  metaFile << "agent_type: Q-Learning\n";
  metaFile << "learning_rate: " << params_.learningRate << "\n";
  metaFile << "discount_factor: " << params_.discountFactor << "\n";
  metaFile << "epsilon: " << epsilon_ << "\n";
  metaFile << "epsilon_min: " << params_.epsilonMin << "\n";
  metaFile << "epsilon_decay: " << params_.epsilonDecay << "\n";
  metaFile << "step_count: " << stepCount_ << "\n";
  metaFile << "state_space_size: " << qTable_.size() << "\n";

  metaFile.close();

  std::cout << "Saved Q-learning agent to " << filepath << "\n";
  std::cout << "  States learned: " << qTable_.size() << "\n";
  std::cout << "  Steps taken: " << stepCount_ << "\n";
  std::cout << "  Current epsilon: " << epsilon_ << "\n";
}

void QLearningAgent::load(const std::string &filepath) {
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
    }
  }

  metaFile.close();

  std::cout << "Loaded Q-learning agent from " << filepath << "\n";
  std::cout << "  States learned: " << qTable_.size() << "\n";
  std::cout << "  Steps taken: " << stepCount_ << "\n";
  std::cout << "  Current epsilon: " << epsilon_ << "\n";
}

void QLearningAgent::reset() {
  qTable_.clear();
  epsilon_ = params_.epsilon;
  stepCount_ = 0;
}

Action QLearningAgent::epsilonGreedy(const State &state,
                                     const std::vector<Action> &validActions) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double rand = dist(rng_);

  if (rand < epsilon_) {
    std::uniform_int_distribution<size_t> actionDist(0,
                                                     validActions.size() - 1);
    return validActions[actionDist(rng_)];
  } else {
    return greedyAction(state, validActions);
  }
}

Action
QLearningAgent::greedyAction(const State &state,
                             const std::vector<Action> &validActions) const {
  return qTable_.getMaxAction(state, validActions);
}

void QLearningAgent::decayEpsilon() {
  epsilon_ *= params_.epsilonDecay;
  epsilon_ = std::max(epsilon_, params_.epsilonMin);
}
} // namespace ai
} // namespace blackjack