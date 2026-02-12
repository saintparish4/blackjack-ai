#include "ai/MonteCarloAgent.hpp" 
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace blackjack {
namespace ai {

    MonteCarloAgent::MonteCarloAgent(const Hyperparameters& params) 
    : params_(params), 
    qTable_(0.0), 
    epsilon_(params.epsilon),
    rng_(std::random_device{}()),
    episodeCount_(0) 
    {
        if (!params_.isValid()) {
            throw std::invalid_argument("Invalid hyperparameters"); 
        }
    }

    Action MonteCarloAgent::chooseAction(
        const State& state, 
        const std::vector<Action>& validActions, 
        bool training 
    ) {
        if (validActions.empty()) {
            throw std::invalid_argument("No valid actions provided"); 
        }

        Action action; 

        if (training) {
            action = epsilonGreedy(state, validActions);
        } else {
            action = greedyAction(state, validActions); 
        }

        // Record state-action pair for current episode 
        if (training) {
            StateActionPair sa{state, action, currentEpisode_.size()}; 
            currentEpisode_.push_back(sa);  
        }

        return action; 
    }

    void MonteCarloAgent::learn(const Experience& experience) {
        // Monte Carlo learns at end of episode, not per step 
        // This method is called for compatability but does nothing 
        // Use finishEpisode instead 
    }

    void MonteCarloAgent::startEpisode() {
        currentEpisode_.clear(); 
    }
}
}