#include "Trainer.hpp"
#include "../ai/GameStateConverter.hpp"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <thread>


namespace blackjack {
namespace training {
Trainer::Trainer(std::shared_ptr<ai::Agent> agent, const TrainingConfig &config)
    : agent_(agent), config_(config),
      game_(std::make_unique<BlackjackGame>(config.gameRules)),
      evaluator_(std::make_unique<Evaluator>(config.gameRules)),
      logger_(std::make_unique<Logger>(config.logDir)), paused_(false),
      shouldStop_(false), episodesSinceImprovement_(0), bestWinRate_(0.0) {
  // Create checkpoint directory if it doesn't exist
  std::filesystem::create_directories(config_.checkpointDir);
  std::filesystem::create_directories(config_.logDir);

  if (config_.verbose) {
    std::cout << "=== Training Configuration ===\n";
    std::cout << "Episodes: " << config_.numEpisodes << "\n";
    std::cout << "Eval frequency: " << config_.evalFrequency << "\n";
    std::cout << "Checkpoint frequency: " << config_.checkpointFrequency
              << "\n";
    std::cout << "Checkpoint dir: " << config_.checkpointDir << "\n";
    std::cout << "Log dir: " << config_.logDir << "\n";
    std::cout << "============================\n\n";
  }
}

TrainingMetrics Trainer::train() { return trainEpisodes(config_.numEpisodes); }

TrainingMetrics Trainer::trainEpisodes(size_t numEpisodes) {
  if (config_.verbose) {
    std::cout << "Starting training for " << numEpisodes << " episodes...\n";
  }

  size_t startEpisode = currentMetrics_.totalEpisodes;
  size_t endEpisode = startEpisode + numEpisodes;

  // Training loop
  for (size_t episode = startEpisode; episode < endEpisode; ++episode) {
    // Check for stop request (signal handler)
    if (shouldStop_) {
      if (config_.verbose) {
        std::cout << "\nStop requested at episode " << (episode + 1)
                  << ". Saving checkpoint...\n";
      }
      saveCheckpoint(episode);
      break;
    }

    // Check for pause
    while (paused_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Run episode
    EpisodeStats stats = runEpisode();
    stats.episodeNumber = episode + 1;

    // Update metrics
    updateMetrics(stats);
    currentMetrics_.totalEpisodes = episode + 1;

    // Periodic evaluation
    if ((episode + 1) % config_.evalFrequency == 0) {
      evaluate();

      // Progress callback
      if (progressCallback_) {
        progressCallback_(currentMetrics_);
      }

      // Early stopping check
      if (shouldStopEarly()) {
        if (config_.verbose) {
          std::cout << "\nEarly stopping triggered at episode " << (episode + 1)
                    << "\n";
        }
        break;
      }
    }

    // Periodic checkpoint
    if ((episode + 1) % config_.checkpointFrequency == 0) {
      saveCheckpoint(episode + 1);
    }
  }

  // Final evaluation
  if (config_.verbose) {
    std::cout << "\nRunning final evaluation...\n";
  }
  evaluate();

  // Final checkpoint
  saveCheckpoint(currentMetrics_.totalEpisodes);

  if (config_.verbose) {
    std::cout << "\n=== Training Complete ===\n";
    std::cout << "Total episodes: " << currentMetrics_.totalEpisodes << "\n";
    std::cout << "Final win rate: " << (currentMetrics_.winRate * 100) << "%\n";
    std::cout << "States learned: " << currentMetrics_.statesLearned << "\n";
    std::cout << "========================\n";
  }

  return currentMetrics_;
}

EpisodeStats Trainer::runEpisode() {
  EpisodeStats stats;
  std::vector<ai::Experience> experiences;

  // Start new round
  game_->startRound();

  // Check for immediate blackjack
  if (game_->isRoundComplete()) {
    const std::vector<Outcome> &outcomes = game_->getOutcomes();
    const std::vector<bool> &wasDoubled = game_->getWasDoubledByHand();
    finishEpisode(experiences, outcomes, wasDoubled);
    stats.outcome = outcomes.empty() ? Outcome::PUSH : outcomes[0];
    stats.reward = 0.0;
    for (size_t i = 0; i < outcomes.size(); ++i) {
      stats.reward += ai::GameStateConverter::outcomeToReward(
          outcomes[i], i < wasDoubled.size() && wasDoubled[i]);
    }
    stats.handsPlayed = 0;
    return stats;
  }

  // Play agent's turn
  playAgentTurn(experiences);

  // Get outcomes (one per hand; multiple after split)
  const std::vector<Outcome> &outcomes = game_->getOutcomes();
  const std::vector<bool> &wasDoubled = game_->getWasDoubledByHand();
  stats.outcome = outcomes.empty() ? Outcome::PUSH : outcomes[0];
  stats.reward = 0.0;
  for (size_t i = 0; i < outcomes.size(); ++i) {
    stats.reward += ai::GameStateConverter::outcomeToReward(
        outcomes[i], i < wasDoubled.size() && wasDoubled[i]);
  }
  stats.handsPlayed = static_cast<int>(experiences.size());
  stats.playerBusted = std::any_of(
      outcomes.begin(), outcomes.end(),
      [](Outcome o) { return o == Outcome::PLAYER_BUST; });
  stats.dealerBusted = std::any_of(
      outcomes.begin(), outcomes.end(),
      [](Outcome o) { return o == Outcome::DEALER_BUST; });

  // Learn from all experiences
  finishEpisode(experiences, outcomes, wasDoubled);

  return stats;
}

void Trainer::playAgentTurn(std::vector<ai::Experience> &experiences) {
  while (!game_->isRoundComplete()) {
    const Hand &playerHand = game_->getPlayerHand();
    const Hand &dealerHand = game_->getDealerHand(true);

    ai::State currentState = ai::GameStateConverter::toAIState(
        playerHand, dealerHand, game_->canSplit(), game_->canDoubleDown());

    std::vector<ai::Action> validActions =
        ai::GameStateConverter::getValidActions(
            playerHand, game_->canSplit(), game_->canDoubleDown(),
            game_->canSurrender());

    ai::Action action = agent_->chooseAction(currentState, validActions, true);

    ai::GameStateConverter::executeAction(action, *game_);

    ai::State nextState;
    std::vector<ai::Action> nextValidActions;
    if (!game_->isRoundComplete()) {
      nextState = ai::GameStateConverter::toAIState(
          game_->getPlayerHand(), game_->getDealerHand(true),
          game_->canSplit(), game_->canDoubleDown());
      nextValidActions = ai::GameStateConverter::getValidActions(
          game_->getPlayerHand(), game_->canSplit(), game_->canDoubleDown(),
          game_->canSurrender());
    }

    experiences.emplace_back(currentState, action, 0.0, nextState,
                             game_->isRoundComplete(),
                             std::move(nextValidActions));
  }
}

void Trainer::finishEpisode(std::vector<ai::Experience> &experiences,
                            const std::vector<Outcome> &outcomes,
                            const std::vector<bool> &wasDoubledByHand) {
  double finalReward = 0.0;
  for (size_t i = 0; i < outcomes.size(); ++i) {
    finalReward += ai::GameStateConverter::outcomeToReward(
        outcomes[i], i < wasDoubledByHand.size() && wasDoubledByHand[i]);
  }

  for (size_t i = 0; i < experiences.size(); ++i) {
    experiences[i].reward = (i + 1 == experiences.size()) ? finalReward : 0.0;
  }

  for (const auto &exp : experiences) {
    agent_->learn(exp);
  }
}

void Trainer::updateMetrics(const EpisodeStats &stats) {
  // Update running averages using exponential moving average
  const double alpha = 0.01; // Smoothing factor

  currentMetrics_.avgReward =
      alpha * stats.reward + (1 - alpha) * currentMetrics_.avgReward;

  // Update counts
  if (stats.outcome == Outcome::PLAYER_WIN ||
      stats.outcome == Outcome::PLAYER_BLACKJACK ||
      stats.outcome == Outcome::DEALER_BUST) {
    currentMetrics_.winRate =
        alpha * 1.0 + (1 - alpha) * currentMetrics_.winRate;
  } else if (stats.outcome == Outcome::PUSH) {
    currentMetrics_.pushRate =
        alpha * 1.0 + (1 - alpha) * currentMetrics_.pushRate;
  } else {
    currentMetrics_.lossRate =
        alpha * 1.0 + (1 - alpha) * currentMetrics_.lossRate;
  }

  if (stats.playerBusted) {
    currentMetrics_.bustRate =
        alpha * 1.0 + (1 - alpha) * currentMetrics_.bustRate;
  }
}

void Trainer::evaluate() {
  if (config_.verbose) {
    std::cout << "\n--- Evaluation at episode " << currentMetrics_.totalEpisodes
              << " ---\n";
  }

  // Run evaluation
  EvaluationResult result =
      evaluator_->evaluate(agent_.get(), config_.evalGames);

  // Update metrics
  currentMetrics_.winRate = result.winRate;
  currentMetrics_.lossRate = result.lossRate;
  currentMetrics_.pushRate = result.pushRate;
  currentMetrics_.avgReward = result.avgReward;
  currentMetrics_.bustRate = result.bustRate;

  // Get exploration metrics via agent interface
  currentMetrics_.currentEpsilon = agent_->getExplorationRate();
  currentMetrics_.statesLearned = agent_->getStateCount();

  // Log metrics
  logger_->log(currentMetrics_);

  // Add to history
  trainingHistory_.push_back(currentMetrics_);

  // Check for improvement
  if (result.winRate > bestWinRate_ + config_.minImprovement) {
    bestWinRate_ = result.winRate;
    episodesSinceImprovement_ = 0;
  } else {
    episodesSinceImprovement_++;
  }

  if (config_.verbose) {
    std::cout << "  Win rate: " << std::fixed << std::setprecision(2)
              << (result.winRate * 100) << "%\n";
    std::cout << "  Avg reward: " << std::setprecision(4) << result.avgReward
              << "\n";
    std::cout << "  Epsilon: " << currentMetrics_.currentEpsilon << "\n";
    std::cout << "  States learned: " << currentMetrics_.statesLearned << "\n";

    if (result.strategyAccuracy > 0) {
      std::cout << "  Strategy accuracy: " << (result.strategyAccuracy * 100)
                << "%\n";
    }

    std::cout << "  Episodes since improvement: " << episodesSinceImprovement_
              << "\n";
  }
}

void Trainer::saveCheckpoint(size_t episodeNum) {
  std::string filename =
      config_.checkpointDir + "/agent_episode_" + std::to_string(episodeNum);

  agent_->save(filename);

  if (config_.verbose) {
    std::cout << "Checkpoint saved: " << filename << "\n";
  }
}

bool Trainer::shouldStopEarly() const {
  return episodesSinceImprovement_ >= config_.earlyStoppingPatience;
}
} // namespace training
} // namespace blackjack