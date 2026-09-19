// WebAssembly entry point for the browser demo (web/).
//
// Exposes one class, Table, to JavaScript through Embind. The browser plays on
// the same engine, the same trained Q-table and the same basic strategy chart
// as the C++ tools; this file only adapts them to a JSON-in-a-string API, so
// the demo cannot drift from what the README measures.

#include "ai/Explain.hpp"
#include "ai/GameStateConverter.hpp"
#include "ai/QLearningAgent.hpp"
#include "game/BlackjackGame.hpp"
#include "training/Evaluator.hpp"

#include <emscripten/bind.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace blackjack;

namespace {

std::string quote(const std::string &s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

std::string num(double v) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(4) << v;
  return os.str();
}

std::string cardsJson(const Hand &hand) {
  std::string out = "[";
  const auto &cards = hand.getCards();
  for (size_t i = 0; i < cards.size(); ++i) {
    if (i) out += ",";
    out += quote(cards[i].toShortString());
  }
  return out + "]";
}

bool parseAction(const std::string &name, ai::Action &out) {
  for (ai::Action a : {ai::Action::HIT, ai::Action::STAND, ai::Action::DOUBLE,
                       ai::Action::SPLIT, ai::Action::SURRENDER}) {
    if (ai::actionToString(a) == name) {
      out = a;
      return true;
    }
  }
  return false;
}

/** "hard 16 vs T", "soft 18 vs 6", "pair of 8s vs A". */
std::string describe(const ai::State &s) {
  const int d = s.dealerUpCard;
  const std::string dealer = d == 1 ? "A" : d == 10 ? "T" : std::to_string(d);
  if (s.canSplit) {
    const int rank = s.hasUsableAce ? 1 : s.playerTotal / 2;
    const std::string r = rank == 1 ? "A" : rank == 10 ? "T" : std::to_string(rank);
    return "pair of " + r + "s vs " + dealer;
  }
  return std::string(s.hasUsableAce ? "soft " : "hard ") +
         std::to_string(s.playerTotal) + " vs " + dealer;
}

} // namespace

/** One seat at a Vegas Strip table, with the trained agent watching. */
class Table {
public:
  explicit Table(uint32_t seed)
      : rules_(GameRules::vegasStrip()), game_(rules_, seed), book_(rules_) {}

  /** Load a model written by train/experiment (path without extension). */
  bool loadModel(const std::string &path) {
    try {
      agent_.load(path);
      loaded_ = agent_.getStateCount() > 0;
    } catch (const std::exception &) {
      loaded_ = false;
    }
    return loaded_;
  }

  std::string deal() {
    game_.startRound();
    dealt_ = true;
    return state();
  }

  /** Apply the player's action; illegal or unknown actions are ignored. */
  std::string act(const std::string &name) {
    ai::Action action;
    if (dealt_ && !game_.isRoundComplete() && parseAction(name, action)) {
      const auto legal = legalActions();
      if (std::find(legal.begin(), legal.end(), action) != legal.end()) {
        ai::GameStateConverter::executeAction(action, game_);
      }
    }
    return state();
  }

  /** Let the agent make the current decision (greedy, no exploration). */
  std::string aiAct() {
    if (dealt_ && !game_.isRoundComplete()) {
      ai::GameStateConverter::executeAction(
          agent_.chooseAction(currentState(), legalActions(), false), game_);
    }
    return state();
  }

  std::string state() const {
    std::ostringstream o;
    const bool done = !dealt_ || game_.isRoundComplete();
    o << "{\"dealt\":" << (dealt_ ? "true" : "false")
      << ",\"done\":" << (done ? "true" : "false")
      << ",\"modelLoaded\":" << (loaded_ ? "true" : "false")
      << ",\"blackjackPays\":" << num(rules_.blackjackPayout);
    if (!dealt_) {
      o << "}";
      return o.str();
    }

    const Hand dealer = game_.getDealerHand(/*hideHoleCard=*/!done);
    o << ",\"dealer\":{\"cards\":" << cardsJson(dealer)
      << ",\"hidden\":" << (done ? "false" : "true")
      << ",\"total\":" << dealer.getTotal()
      << ",\"soft\":" << (dealer.isSoft() ? "true" : "false") << "}";

    const auto &hands = game_.getPlayerHands();
    const auto &outcomes = game_.getOutcomes();
    const auto &doubled = game_.getWasDoubledByHand();
    double roundReward = 0.0;
    o << ",\"hands\":[";
    for (size_t i = 0; i < hands.size(); ++i) {
      if (i) o << ",";
      const bool wasDoubled = i < doubled.size() && doubled[i];
      o << "{\"cards\":" << cardsJson(hands[i]) << ",\"total\":" << hands[i].getTotal()
        << ",\"soft\":" << (hands[i].isSoft() ? "true" : "false")
        << ",\"doubled\":" << (wasDoubled ? "true" : "false");
      if (done && i < outcomes.size()) {
        const double reward = ai::GameStateConverter::outcomeToReward(
            outcomes[i], wasDoubled, rules_.blackjackPayout);
        roundReward += reward;
        o << ",\"outcome\":" << quote(outcomeToString(outcomes[i]))
          << ",\"reward\":" << num(reward);
      }
      o << "}";
    }
    o << "],\"current\":" << game_.getCurrentHandIndex();

    if (done) {
      o << ",\"roundReward\":" << num(roundReward) << "}";
      return o.str();
    }

    const ai::State s = currentState();
    const auto legal = legalActions();
    o << ",\"legal\":[";
    for (size_t i = 0; i < legal.size(); ++i) {
      if (i) o << ",";
      o << quote(ai::actionToString(legal[i]));
    }
    o << "]";

    // The agent's view of this decision: Q-value per legal action, whether
    // every one of them was actually trained, its choice and how sure it is.
    const auto q = agent_.getAllQValues(s);
    bool trained = loaded_;
    double best = std::numeric_limits<double>::lowest();
    double second = std::numeric_limits<double>::lowest();
    o << ",\"q\":{";
    for (size_t i = 0; i < legal.size(); ++i) {
      const double v = q[static_cast<size_t>(legal[i])];
      trained = trained && agent_.getVisitCount(s, legal[i]) > 0;
      if (v > best) { second = best; best = v; }
      else if (v > second) { second = v; }
      if (i) o << ",";
      o << quote(ai::actionToString(legal[i])) << ":" << num(v);
    }
    o << "}";
    const double margin = second == std::numeric_limits<double>::lowest() ? 0.0 : best - second;
    const ai::Action aiAction = agent_.chooseAction(s, legal, false);

    // The book's play, as the audit reads the chart: DOUBLE falls back when
    // the hand can't double, SURRENDER only when the rules offer it.
    ai::Action bookAction = book_.getAction(s, rules_.surrender);
    if (bookAction == ai::Action::DOUBLE && !s.canDouble) {
      bookAction = book_.fallbackWhenCannotDouble(s);
    }

    o << ",\"situation\":" << quote(describe(s))
      << ",\"trained\":" << (trained ? "true" : "false")
      << ",\"ai\":" << quote(ai::actionToString(aiAction))
      << ",\"margin\":" << num(margin)
      << ",\"confidence\":" << quote(ai::confidenceLabel(margin))
      << ",\"reason\":" << quote(ai::actionReason(aiAction, s))
      << ",\"book\":" << quote(ai::actionToString(bookAction))
      << ",\"bookReason\":" << quote(ai::actionReason(bookAction, s)) << "}";
    return o.str();
  }

private:
  GameRules rules_;
  BlackjackGame game_;
  training::BasicStrategy book_;
  // Agent::chooseAction isn't const because training-mode choices advance the
  // RNG; the greedy choices made here never do.
  mutable ai::QLearningAgent agent_;
  bool loaded_ = false;
  bool dealt_ = false;

  ai::State currentState() const {
    return ai::GameStateConverter::toAIState(
        game_.getPlayerHand(), game_.getDealerHand(true), game_.canSplit(),
        game_.canDoubleDown());
  }

  std::vector<ai::Action> legalActions() const {
    return ai::GameStateConverter::getValidActions(
        game_.getPlayerHand(), game_.canSplit(), game_.canDoubleDown(),
        game_.canSurrender());
  }
};

EMSCRIPTEN_BINDINGS(blackjack) {
  emscripten::class_<Table>("Table")
      .constructor<uint32_t>()
      .function("loadModel", &Table::loadModel)
      .function("deal", &Table::deal)
      .function("act", &Table::act)
      .function("aiAct", &Table::aiAct)
      .function("state", &Table::state);
}
