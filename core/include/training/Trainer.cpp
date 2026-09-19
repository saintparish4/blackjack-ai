#include "Trainer.hpp"
#include "../ai/GameStateConverter.hpp"
#include "../util/ProgressBar.hpp"
#include "../util/Seed.hpp"
#include "ConvergenceReport.hpp"
#include "StrategyChart.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>


namespace blackjack {
namespace training {

EarlyStopMetric earlyStopMetricFromString(const std::string &name) {
  if (name == "accuracy") return EarlyStopMetric::ACCURACY;
  if (name == "ev") return EarlyStopMetric::EV;
  throw std::invalid_argument("early_stopping_metric must be 'accuracy' or "
                              "'ev', got '" + name + "'");
}

std::string toString(EarlyStopMetric metric) {
  return metric == EarlyStopMetric::EV ? "ev" : "accuracy";
}

namespace {

std::optional<uint32_t> streamSeed(const std::optional<uint32_t> &seed,
                                   util::SeedStream stream) {
  if (!seed) return std::nullopt;
  return util::deriveSeed(*seed, stream);
}

} // namespace

Trainer::Trainer(std::shared_ptr<ai::Agent> agent, const TrainingConfig &config)
    : agent_(agent), config_(config),
      game_(std::make_unique<BlackjackGame>(
          config.gameRules, streamSeed(config.seed, util::SeedStream::TRAIN_DEAL))),
      evaluator_(std::make_unique<Evaluator>(
          config.gameRules, streamSeed(config.seed, util::SeedStream::EVAL_DEAL))),
      logger_(config.logDir.empty() ? nullptr
                                    : std::make_unique<Logger>(config.logDir)),
      paused_(false),
      shouldStop_(false), episodesSinceImprovement_(0),
      bestMetric_(std::numeric_limits<double>::lowest()) {
  if (config_.saveCheckpoints) {
    std::filesystem::create_directories(config_.checkpointDir);
  }

  if (config_.verbose) {
    std::cout << "=== Training Configuration ===\n";
    std::cout << "Episodes: " << config_.numEpisodes << "\n";
    std::cout << "Eval frequency: " << config_.evalFrequency << "\n";
    std::cout << "Checkpoint frequency: " << config_.checkpointFrequency
              << "\n";
    std::cout << "Checkpoint dir: " << config_.checkpointDir << "\n";
    std::cout << "Log dir: " << config_.logDir << "\n";
    std::cout << "Seed: "
              << (config_.seed ? std::to_string(*config_.seed) : "none (nondeterministic)")
              << "\n";
    std::cout << "Early stopping: " << toString(config_.earlyStoppingMetric)
              << ", patience " << config_.earlyStoppingPatience << "\n";
    std::cout << "============================\n\n";
  }
}

TrainingMetrics Trainer::train() {
  TrainingMetrics metrics = trainEpisodes(config_.numEpisodes);
  runAndSaveReport(metrics);
  return metrics;
}

TrainingMetrics Trainer::trainEpisodes(size_t numEpisodes) {
  if (config_.verbose) {
    std::cout << "Starting training for " << numEpisodes << " episodes...\n";
  }

  const auto loopStart = std::chrono::steady_clock::now();
  size_t startEpisode = currentMetrics_.totalEpisodes;
  size_t endEpisode = startEpisode + numEpisodes;

  util::ProgressBar progressBar(numEpisodes, 1000);
  if (!config_.verbose) progressBar.setSilent(true);

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
    if (config_.saveCheckpoints && config_.checkpointFrequency > 0 &&
        (episode + 1) % config_.checkpointFrequency == 0) {
      saveCheckpoint(episode + 1);
    }

    // Progress bar update (info only on eval episodes when metrics are fresh)
    std::string info;
    if ((episode + 1) % config_.evalFrequency == 0) {
      info = "Win: " +
             std::to_string(static_cast<int>(currentMetrics_.winRate * 100)) +
             "%" + " | eps: " +
             std::to_string(currentMetrics_.currentEpsilon).substr(0, 5);
    }
    progressBar.update(episode + 1 - startEpisode, info);
  }

  // A bet-learning agent measures its own edge at each count, playing
  // greedily on a separate shoe, before anything evaluates it.
  if (agent_->learnsBets() && config_.betCalibrationRounds > 0) {
    if (config_.verbose) {
      std::cout << "\nCalibrating bets over " << config_.betCalibrationRounds
                << " greedy rounds...\n";
    }
    Evaluator calibrator(config_.gameRules,
                         streamSeed(config_.seed, util::SeedStream::BET_CALIBRATION));
    calibrator.calibrateBets(agent_.get(), config_.betCalibrationRounds);
  }

  // Final evaluation
  if (config_.verbose) {
    std::cout << "\nRunning final evaluation...\n";
  }
  evaluate();
  progressBar.finish("Done");

  // Final checkpoint
  saveCheckpoint(currentMetrics_.totalEpisodes);
  trainingSeconds_ += std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - loopStart)
                          .count();

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
    finishEpisode(experiences, {}, outcomes, wasDoubled);
    stats.outcome = outcomes.empty() ? Outcome::PUSH : outcomes[0];
    stats.reward = 0.0;
    for (size_t i = 0; i < outcomes.size(); ++i) {
      stats.reward += ai::GameStateConverter::outcomeToReward(
          outcomes[i], i < wasDoubled.size() && wasDoubled[i],
          config_.gameRules.blackjackPayout);
    }
    stats.handsPlayed = 0;
    return stats;
  }

  // Play agent's turn
  std::vector<StepInfo> steps;
  playAgentTurn(experiences, steps);

  // Get outcomes (one per hand; multiple after split)
  const std::vector<Outcome> &outcomes = game_->getOutcomes();
  const std::vector<bool> &wasDoubled = game_->getWasDoubledByHand();
  stats.outcome = outcomes.empty() ? Outcome::PUSH : outcomes[0];
  stats.reward = 0.0;
  for (size_t i = 0; i < outcomes.size(); ++i) {
    stats.reward += ai::GameStateConverter::outcomeToReward(
        outcomes[i], i < wasDoubled.size() && wasDoubled[i],
        config_.gameRules.blackjackPayout);
  }
  stats.handsPlayed = static_cast<int>(experiences.size());
  stats.playerBusted = std::any_of(
      outcomes.begin(), outcomes.end(),
      [](Outcome o) { return o == Outcome::PLAYER_BUST; });
  stats.dealerBusted = std::any_of(
      outcomes.begin(), outcomes.end(),
      [](Outcome o) { return o == Outcome::DEALER_BUST; });

  // Learn from all experiences
  finishEpisode(experiences, steps, outcomes, wasDoubled);

  return stats;
}

void Trainer::playAgentTurn(std::vector<ai::Experience> &experiences,
                            std::vector<StepInfo> &steps) {
  while (!game_->isRoundComplete()) {
    const Hand &playerHand = game_->getPlayerHand();
    const Hand &dealerHand = game_->getDealerHand(true);

    ai::State currentState = ai::GameStateConverter::toAIState(
        playerHand, dealerHand, game_->canSplit(), game_->canDoubleDown(),
        ai::GameStateConverter::countBucketFor(*agent_, *game_));

    std::vector<ai::Action> validActions =
        ai::GameStateConverter::getValidActions(
            playerHand, game_->canSplit(), game_->canDoubleDown(),
            game_->canSurrender());

    ai::Action action = agent_->chooseAction(currentState, validActions, true);

    const size_t handBefore = game_->getCurrentHandIndex();
    const size_t handsBefore = game_->getPlayerHands().size();
    ai::GameStateConverter::executeAction(action, *game_);

    StepInfo step;
    step.hand = handBefore;
    step.isSplit = game_->getPlayerHands().size() > handsBefore;
    // Finishing the first split hand moves play to the second; that must
    // not read as a transition into the second hand's state.
    step.endsHand = !step.isSplit &&
                    (game_->isRoundComplete() ||
                     game_->getCurrentHandIndex() != handBefore);
    steps.push_back(step);

    ai::State nextState;
    std::vector<ai::Action> nextValidActions;
    if (!game_->isRoundComplete()) {
      nextState = ai::GameStateConverter::toAIState(
          game_->getPlayerHand(), game_->getDealerHand(true),
          game_->canSplit(), game_->canDoubleDown(),
          ai::GameStateConverter::countBucketFor(*agent_, *game_));
      nextValidActions = ai::GameStateConverter::getValidActions(
          game_->getPlayerHand(), game_->canSplit(), game_->canDoubleDown(),
          game_->canSurrender());
    }

    experiences.emplace_back(currentState, action, 0.0, nextState,
                             game_->isRoundComplete(),
                             std::move(nextValidActions));

    if (step.isSplit && !game_->isRoundComplete()) {
      // The second hand is dealt at the split but played later. The engine
      // allows neither a resplit nor a double after splitting, so when play
      // reaches it the options are the ones the first hand has now; the
      // SplitCreditsEachHandSeparately test holds this in place.
      const Hand &second = game_->getPlayerHands()[1];
      experiences.back().secondNextState = ai::GameStateConverter::toAIState(
          second, game_->getDealerHand(true), game_->canSplit(),
          game_->canDoubleDown(),
          ai::GameStateConverter::countBucketFor(*agent_, *game_));
      experiences.back().secondValidNextActions =
          ai::GameStateConverter::getValidActions(second, game_->canSplit(),
                                                  game_->canDoubleDown(),
                                                  game_->canSurrender());
    }
  }
}

void Trainer::finishEpisode(std::vector<ai::Experience> &experiences,
                            const std::vector<StepInfo> &steps,
                            const std::vector<Outcome> &outcomes,
                            const std::vector<bool> &wasDoubledByHand) {
  std::vector<double> handReward(outcomes.size(), 0.0);
  double total = 0.0;
  for (size_t i = 0; i < outcomes.size(); ++i) {
    handReward[i] = ai::GameStateConverter::outcomeToReward(
        outcomes[i], i < wasDoubledByHand.size() && wasDoubledByHand[i],
        config_.gameRules.blackjackPayout);
    total += handReward[i];
  }

  // Split rounds used to flow as one chain: SPLIT bootstrapped from the first
  // hand alone, the first hand's last decision bootstrapped from the second
  // hand's state, and the summed reward landed on the second hand's last
  // decision. A split then looked worth one hand, so splitting a losing pair
  // (7,7 against a T) looked half as bad as it is and the agent over-split.
  //
  // Now SPLIT bootstraps from both hands' states (see playAgentTurn), and each
  // hand's last decision is terminal with that hand's own reward. A Monte
  // Carlo total for SPLIT was tried first; it values the split under the
  // exploring play that follows it, mostly from early training, and swung the
  // agent to under-splitting instead.
  for (size_t i = 0; i < experiences.size(); ++i) {
    ai::Experience &exp = experiences[i];
    const StepInfo &step = steps[i];
    if (step.isSplit) {
      // Split aces settle on the spot, so their split is terminal.
      exp.reward = exp.done ? total : 0.0;
    } else if (step.endsHand) {
      exp.done = true;
      exp.reward = step.hand < handReward.size() ? handReward[step.hand] : total;
    } else {
      exp.reward = 0.0;
    }
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
  currentMetrics_.strategyAccuracy = result.strategyAccuracy;

  // Get exploration metrics via agent interface
  currentMetrics_.currentEpsilon = agent_->getExplorationRate();
  currentMetrics_.statesLearned = agent_->getStateCount();

  // Log metrics
  if (logger_) {
    logger_->log(currentMetrics_);
  }

  // Add to history
  trainingHistory_.push_back(currentMetrics_);

  // Check for improvement. Accuracy comes from an exhaustive audit of the
  // greedy policy, so it only moves when the policy does; EV is a sample mean
  // and needs evalGames large enough that noise does not reset patience.
  const double metric = config_.earlyStoppingMetric == EarlyStopMetric::EV
                            ? result.avgReward
                            : result.strategyAccuracy;
  if (metric > bestMetric_ + config_.minImprovement) {
    bestMetric_ = metric;
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
  if (!config_.saveCheckpoints) {
    return;
  }
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

// ---- helpers for runAndSaveReport -------------------------------------------

namespace {

/** Emit improvement suggestions based on convergence quality and final metrics. */
void writeSuggestions(const ConvergenceResult &cr, const TrainingMetrics &m,
                      const BaselineComparison &cmp, std::ostream &out) {
  out << "\n=== Improvement Suggestions ===\n";
  bool any = false;

  if (!cr.passed) {
    out << "  • Strategy accuracy ("
        << std::fixed << std::setprecision(1) << (cr.accuracy * 100)
        << "%) is below the 90% threshold.\n"
        << "    Try training for more episodes (e.g. 2M+) or reduce epsilon_decay.\n";
    any = true;
  }

  size_t critCount = 0;
  size_t softDivCount = 0;
  for (const auto &d : cr.divergences) {
    if (d.isCritical)       ++critCount;
    if (d.state.hasUsableAce) ++softDivCount;
  }

  if (critCount > 0) {
    out << "  • " << critCount << " critical state(s) diverge from basic strategy.\n"
        << "    High-stakes hands (hard 16 vs strong dealer, hard 10/11) need more exploration.\n";
    any = true;
  }

  if (!cr.divergences.empty() &&
      softDivCount > cr.divergences.size() / 2) {
    out << "  • Soft-total strategy shows above-average divergences ("
        << softDivCount << " states).\n"
        << "    Ace-involved hands are rare; extended training usually resolves these.\n";
    any = true;
  }

  // Judged on EV with its interval, not on win rate: win rate saturates near
  // 43% for almost any sensible policy, and the last training-loop evaluation
  // is only eval_games rounds, so it swings by several points on noise alone.
  const double agentHigh = cmp.agent.avgReward + 1.96 * cmp.agent.rewardStdErr;
  const double bookLow = cmp.book.avgReward - 1.96 * cmp.book.rewardStdErr;
  if (cmp.agent.gamesPlayed > 0 && agentHigh < bookLow) {
    out << "  • EV per round (" << std::fixed << std::setprecision(4)
        << cmp.agent.avgReward << ") is below basic strategy ("
        << cmp.book.avgReward << ") and the 95% intervals do not overlap,\n"
        << "    so the gap is real. The divergences above show where it leaks.\n";
    any = true;
  }

  if (m.statesLearned < 150) {
    out << "  • Only " << m.statesLearned
        << " states explored — Q-table is underpopulated.\n"
        << "    Slow epsilon decay (e.g. 0.9999) allows broader exploration.\n";
    any = true;
  }

  if (!any) {
    out << "  • Agent closely matches basic strategy and win rate looks healthy.\n"
        << "    No major issues detected — consider running a longer eval for confidence.\n";
  }
  out << "================================\n";
}

} // anonymous namespace

void Trainer::runAndSaveReport(const TrainingMetrics &finalMetrics) {
  // Compute convergence once; reuse for both terminal output and file report.
  ConvergenceReport convergenceReport;
  ConvergenceResult cr = convergenceReport.analyze(
      *agent_, evaluator_->getBasicStrategy(), config_.gameRules.surrender);

  // Baselines on the same engine, rules and evaluator as the agent. Cell-match
  // accuracy over-weights states that barely occur; EV is what the game
  // actually pays out on, and it is only interpretable next to a reference.
  //
  // The agent is re-evaluated here over the same number of rounds as the
  // baselines. Reusing the training-loop figure would compare a few thousand
  // rounds against two hundred thousand: per-round reward has a standard
  // deviation near 1.15, so a 2,000-round mean carries a +/-0.05 error bar and
  // can land anywhere from "twice as bad as random-ish" to "better than the
  // book" on noise alone.
  //
  // A seeded run measures on the same dedicated stream as the experiment
  // runner, so `train --seed N` and seed N of an experiment report identical
  // numbers for the identical agent they train.
  const size_t baselineGames = config_.reportEvalRounds;
  Evaluator seededFinal(config_.gameRules,
                        streamSeed(config_.seed, util::SeedStream::FINAL_EVAL));
  Evaluator &finalEvaluator = config_.seed ? seededFinal : *evaluator_;
  BaselineComparison cmp = finalEvaluator.compareAgainstBaselines(
      agent_.get(), baselineGames,
      config_.seed ? util::deriveSeed(*config_.seed, util::SeedStream::BASELINE)
                   : 1234);
  const EvaluationResult &agentResult = cmp.agent;
  const EvaluationResult &bookResult = cmp.book;
  const EvaluationResult &randomResult = cmp.random;

  // ----- Terminal output (colored strategy chart + convergence) -----
  if (config_.verbose) {
    StrategyChart chart;
    chart.print(*agent_, evaluator_->getBasicStrategy(), std::cout,
                /*forceNoColor=*/false, config_.gameRules.surrender);
    convergenceReport.print(cr, std::cout);
    writeSuggestions(cr, finalMetrics, cmp, std::cout);
  }

  // ----- Build plain-text report for file -----
  // Fractional seconds: a whole-second count rendered a 1.7s run as
  // "1 seconds" and then divided by it, overstating throughput by 70%.
  const double durationSec = trainingSeconds_;

  // Hyperparameters print at up to 10 significant digits: the stream default
  // of 6 turns an epsilon_decay of 0.9999985 into 0.999999.
  auto param = [](double v) {
    std::ostringstream p;
    p << std::defaultfloat << std::setprecision(10) << v;
    return p.str();
  };

  std::ostringstream oss;

  oss << "============================================================\n"
      << "            Blackjack AI Training Report\n"
      << "============================================================\n\n";

  // 1. Configuration
  oss << "--- Configuration ---\n";
  oss << std::left << std::setw(24) << "Rules preset"
      << ": " << config_.rulesPresetName   << "\n";
  oss << std::setw(24) << "Num decks"
      << ": " << config_.gameRules.numDecks << "\n";
  oss << std::setw(24) << "Dealer hits soft 17"
      << ": " << (config_.gameRules.dealerHitsSoft17 ? "yes" : "no") << "\n";
  oss << std::setw(24) << "Surrender enabled"
      << ": " << (config_.gameRules.surrender ? "yes" : "no") << "\n";
  oss << std::setw(24) << "Blackjack payout"
      << ": " << param(config_.gameRules.blackjackPayout) << ":1\n";
  oss << std::setw(24) << "Agent"
      << ": " << config_.agentName << "\n";
  oss << std::setw(24) << "Learning rate"
      << ": " << param(config_.learningRate)   << "\n";
  oss << std::setw(24) << "Discount factor"
      << ": " << param(config_.discountFactor) << "\n";
  oss << std::setw(24) << "Epsilon start"
      << ": " << param(config_.epsilon)        << "\n";
  oss << std::setw(24) << "Epsilon decay"
      << ": " << param(config_.epsilonDecay)   << "\n";
  oss << std::setw(24) << "Epsilon min"
      << ": " << param(config_.epsilonMin)     << "\n";
  oss << std::setw(24) << "Seed"
      << ": " << (config_.seed ? std::to_string(*config_.seed) : "none") << "\n";
  oss << std::setw(24) << "Early stopping"
      << ": " << toString(config_.earlyStoppingMetric) << ", patience "
      << config_.earlyStoppingPatience << "\n";
  oss << std::setw(24) << "Eval frequency"
      << ": " << config_.evalFrequency  << " episodes\n";
  oss << std::setw(24) << "Eval games"
      << ": " << config_.evalGames      << "\n";

  // 2. Training stats
  oss << "\n--- Training Stats ---\n";
  oss << std::setw(24) << "Total episodes"
      << ": " << finalMetrics.totalEpisodes << "\n";
  oss << std::setw(24) << "Training time"
      << ": " << std::fixed << std::setprecision(2) << durationSec
      << " s (includes periodic evaluations)\n";
  if (durationSec > 0) {
    oss << std::setw(24) << "Episodes / sec"
        << ": " << std::setprecision(0)
        << (static_cast<double>(finalMetrics.totalEpisodes) / durationSec) << "\n";
  }
  oss << std::setw(24) << "States learned"
      << ": " << finalMetrics.statesLearned << "\n";
  oss << std::setw(24) << "Final epsilon"
      << ": " << std::fixed << std::setprecision(6)
      << finalMetrics.currentEpsilon << "\n";

  // 3. Final metrics
  oss << "\n--- Final Performance ---\n";
  oss << std::fixed << std::setprecision(2);
  oss << std::setw(24) << "Win rate"  << ": " << (finalMetrics.winRate  * 100) << "%\n";
  oss << std::setw(24) << "Loss rate" << ": " << (finalMetrics.lossRate * 100) << "%\n";
  oss << std::setw(24) << "Push rate" << ": " << (finalMetrics.pushRate * 100) << "%\n";
  oss << std::setw(24) << "Bust rate" << ": " << (finalMetrics.bustRate * 100) << "%\n";
  oss << std::setw(24) << "Avg reward"<< ": " << std::setprecision(4)
      << finalMetrics.avgReward << "\n";

  // 3b. EV against baselines measured on the same engine and rules.
  oss << "\n--- Expected Value (per round, " << baselineGames
      << " rounds per baseline) ---\n";
  oss << std::fixed << std::setprecision(4);
  oss << std::setw(24) << "Agent" << ": " << agentResult.avgReward
      << " +/- " << (1.96 * agentResult.rewardStdErr) << " (95% CI)\n";
  oss << std::setw(24) << "Basic strategy" << ": " << bookResult.avgReward
      << " +/- " << (1.96 * bookResult.rewardStdErr) << " (95% CI)\n";
  oss << std::setw(24) << "Random play" << ": " << randomResult.avgReward
      << " +/- " << (1.96 * randomResult.rewardStdErr) << " (95% CI)\n";

  if (!std::isnan(cmp.gapRecovered)) {
    oss << std::setw(24) << "EV gap recovered" << ": "
        << std::setprecision(1) << (cmp.gapRecovered * 100)
        << "% of the random-to-book range\n";
  }
  if (config_.gameRules.maxBetUnits > 1) {
    oss << "\n--- Betting (1-" << config_.gameRules.maxBetUnits
        << " units, EV per unit wagered) ---\n";
    oss << std::setw(24) << "Agent" << ": " << std::setprecision(4)
        << agentResult.evPerUnitBet << " +/- " << (1.96 * agentResult.evPerUnitStdErr)
        << " (avg bet " << std::setprecision(2) << agentResult.avgBet << ")\n";
    oss << std::setw(24) << "Basic strategy (flat)" << ": " << std::setprecision(4)
        << bookResult.evPerUnitBet << " +/- " << (1.96 * bookResult.evPerUnitStdErr) << "\n";
    oss << std::setprecision(4);
  }

  // Per settled hand, over the same rounds as the EV line: the "Final
  // Performance" rates above come from the last eval_games-round evaluation
  // and carry a margin of several points.
  oss << std::setw(24) << "Agent win/loss/push" << ": " << std::setprecision(2)
      << (agentResult.winRate * 100) << "% / " << (agentResult.lossRate * 100)
      << "% / " << (agentResult.pushRate * 100) << "% per hand\n";

  // 4. Strategy chart (plain text — uppercase=correct, lowercase=diverges)
  StrategyChart chart;
  chart.print(*agent_, evaluator_->getBasicStrategy(), oss, /*forceNoColor=*/true,
              config_.gameRules.surrender);

  // 5 & 6. Convergence report (includes top-N divergences table)
  convergenceReport.print(cr, oss);

  // 7. Improvement suggestions
  writeSuggestions(cr, finalMetrics, cmp, oss);

  // ----- Write to file -----
  std::filesystem::create_directories(config_.reportDir);
  std::string reportPath = config_.reportDir + "/training_report.txt";
  std::ofstream file(reportPath);
  if (file) {
    file << oss.str();
    std::cout << "\nTraining report saved: " << reportPath << "\n";
  } else {
    std::cerr << "Warning: could not write training report to " << reportPath << "\n";
  }
}
} // namespace training
} // namespace blackjack