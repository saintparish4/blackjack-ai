#include "RunConfig.hpp"
#include "../util/Seed.hpp"
#include <stdexcept>

namespace blackjack {
namespace training {

GameRules rulesFromPreset(const std::string &preset) {
  if (preset == "vegas-strip")   return GameRules::vegasStrip();
  if (preset == "downtown")      return GameRules::downtown();
  if (preset == "atlantic-city") return GameRules::atlanticCity();
  if (preset == "european")      return GameRules::european();
  if (preset == "single-deck")   return GameRules::singleDeck();
  throw std::invalid_argument(
      "unknown rules preset '" + preset +
      "' (expected vegas-strip, downtown, atlantic-city, european, single-deck)");
}

namespace {

size_t nonNegative(const util::ConfigParser &cfg, const std::string &key,
                   int defaultVal) {
  int v = cfg.getInt(key, defaultVal);
  if (v < 0) {
    throw std::invalid_argument("config key '" + key + "' must be >= 0");
  }
  return static_cast<size_t>(v);
}

} // namespace

RunConfig loadRunConfig(const util::ConfigParser &cfg,
                        const RunOverrides &overrides) {
  RunConfig run;
  TrainingConfig &t = run.training;

  t.numEpisodes = nonNegative(cfg, "episodes", 1'000'000);
  if (overrides.episodes) t.numEpisodes = *overrides.episodes;

  t.evalFrequency       = nonNegative(cfg, "eval_frequency", 10'000);
  t.evalGames           = nonNegative(cfg, "eval_games", 1'000);
  t.checkpointFrequency = nonNegative(cfg, "checkpoint_frequency", 50'000);
  t.checkpointDir       = cfg.getString("checkpoint_dir", "./checkpoints");
  t.logDir              = cfg.getString("log_dir", "./logs");
  t.reportDir           = cfg.getString("report_dir", "./analysis");
  t.verbose             = cfg.getBool("verbose", true);
  t.reportEvalRounds    = nonNegative(cfg, "eval_rounds", 200'000);
  if (t.evalFrequency == 0) {
    throw std::invalid_argument("eval_frequency must be > 0");
  }

  t.earlyStoppingPatience = nonNegative(cfg, "early_stopping_patience", 10);
  t.minImprovement        = cfg.getDouble("min_improvement", 0.001);
  t.earlyStoppingMetric   = earlyStopMetricFromString(
      cfg.getString("early_stopping_metric", "accuracy"));

  if (cfg.has("seed")) {
    t.seed = static_cast<uint32_t>(nonNegative(cfg, "seed", 0));
  }
  if (overrides.seed) t.seed = overrides.seed;

  // ---- Agent ----
  run.agentType = cfg.getString("agent", "q-learning");
  if (run.agentType != "q-learning") {
    throw std::invalid_argument("unknown agent '" + run.agentType +
                                "' (expected q-learning)");
  }
  ai::QLearningAgent::Hyperparameters &p = run.agentParams;
  p.learningRate   = cfg.getDouble("learning_rate", 0.1);
  p.discountFactor = cfg.getDouble("discount_factor", 0.95);
  p.epsilon        = cfg.getDouble("epsilon", 1.0);
  p.epsilonDecay   = cfg.getDouble("epsilon_decay", 0.99995);
  p.epsilonMin     = cfg.getDouble("epsilon_min", 0.01);
  if (!p.isValid()) {
    throw std::invalid_argument("invalid agent hyperparameters in config");
  }

  // ---- Rules ----
  run.rulesPreset = cfg.getString("rules_preset", "vegas-strip");
  if (overrides.rulesPreset) run.rulesPreset = *overrides.rulesPreset;
  GameRules rules = rulesFromPreset(run.rulesPreset);

  if (cfg.has("num_decks")) {
    rules.numDecks = nonNegative(cfg, "num_decks", 6);
    if (rules.numDecks == 0) throw std::invalid_argument("num_decks must be >= 1");
  }
  if (cfg.has("dealer_hits_soft_17"))
    rules.dealerHitsSoft17 = cfg.getBool("dealer_hits_soft_17");
  if (cfg.has("surrender"))
    rules.surrender = cfg.getBool("surrender");
  if (cfg.has("blackjack_payout"))
    rules.blackjackPayout = cfg.getDouble("blackjack_payout");
  if (cfg.has("penetration")) {
    rules.penetration = cfg.getDouble("penetration");
    if (rules.penetration <= 0.0 || rules.penetration > 1.0) {
      throw std::invalid_argument("penetration must be in (0, 1]");
    }
  }
  t.gameRules = rules;

  // Reporting copies
  t.rulesPresetName = run.rulesPreset;
  t.agentName       = run.agentType;
  t.learningRate    = p.learningRate;
  t.discountFactor  = p.discountFactor;
  t.epsilon         = p.epsilon;
  t.epsilonDecay    = p.epsilonDecay;
  t.epsilonMin      = p.epsilonMin;

  return run;
}

std::shared_ptr<ai::QLearningAgent> makeAgent(const RunConfig &run,
                                              std::optional<uint32_t> seed) {
  std::optional<uint32_t> agentSeed;
  if (seed) agentSeed = util::deriveSeed(*seed, util::SeedStream::AGENT);
  return std::make_shared<ai::QLearningAgent>(run.agentParams, agentSeed);
}

} // namespace training
} // namespace blackjack
