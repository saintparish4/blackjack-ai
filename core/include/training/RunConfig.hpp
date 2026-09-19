#pragma once

#include "../ai/QLearningAgent.hpp"
#include "../game/GameRules.hpp"
#include "../util/ConfigParser.hpp"
#include "Trainer.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace blackjack {
namespace training {

/**
 * @brief Everything one training run needs, resolved from a config file.
 *
 * train and experiment both build their runs through loadRunConfig, so a
 * config file means the same thing to both. They used to parse it separately,
 * and the CLI-over-config precedence rules had already drifted once.
 */
struct RunConfig {
  TrainingConfig training;
  ai::QLearningAgent::Hyperparameters agentParams;
  std::string agentType = "q-learning";
  std::string rulesPreset = "vegas-strip";
};

/** Values passed explicitly on the command line. Each one, when set, beats the
 *  config file; an unset one never does, whatever the flag's default is. */
struct RunOverrides {
  std::optional<size_t> episodes;
  std::optional<std::string> rulesPreset;
  std::optional<uint32_t> seed;
};

/** Named rule preset. @throws std::invalid_argument on an unknown name: a
 *  typo silently training on default rules produces numbers for the wrong game. */
GameRules rulesFromPreset(const std::string &preset);

/**
 * @brief Resolve a run from a config file plus CLI overrides.
 *
 * Precedence: CLI override > config file > built-in default. Rule fields
 * (num_decks, surrender, ...) apply on top of whichever preset won.
 *
 * @throws std::invalid_argument / std::runtime_error on invalid values.
 */
RunConfig loadRunConfig(const util::ConfigParser &cfg,
                        const RunOverrides &overrides = {});

/** Build the agent a run describes. seed is the run seed; the agent's
 *  exploration stream is derived from it, so it is decorrelated from the
 *  shoes the same seed drives. */
std::shared_ptr<ai::QLearningAgent>
makeAgent(const RunConfig &run, std::optional<uint32_t> seed = std::nullopt);

} // namespace training
} // namespace blackjack
