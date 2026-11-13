#pragma once

/**
 * @file Solver.hpp
 * @brief Exact expected value of a blackjack policy, by combinatorial analysis.
 *
 * Every published number in this project used to be a Monte Carlo estimate
 * graded against a hand-typed chart. This is the ground truth those numbers
 * are now checked against: for a given shoe composition and rules, the exact
 * EV of every action for every hand composition against every upcard, and the
 * exact EV of a whole policy.
 *
 * What it computes, and the conventions that make it match Eric Farmer's CA
 * (github.com/possibly-wrong/blackjack) digit for digit:
 *
 *  - Values are unconditional on the dealer's peek. A dealer natural is a
 *    loss of one bet inside every value; DOUBLE and SPLIT add the extra bet
 *    back, since with a peek a natural takes only the initial wager.
 *  - Hands are compositions. Two orders of the same cards share one memo
 *    entry, and every draw depletes the shoe.
 *  - One split, as the engine allows. A split is worth 2 x the EV of one hand
 *    that starts with the pair card, drawn from a shoe missing the other pair
 *    card (Farmer's SPL1 algorithm). Interaction between the two split hands'
 *    draws is not modelled; training/EngineAudit.hpp bounds its size.
 *  - A Policy decides every action. OptimalPolicy is CDZ-: the best play for
 *    each composition, with the unsplit choice reused after a split.
 *    AgentPolicy plays any ai::Agent through the same State it sees in the
 *    engine, so the exact EV of the book chart or a learned Q-table is one
 *    call away.
 *
 * Invariant: nothing here samples. Same rules, shoe and policy, same answer.
 */

#include "../ai/Agent.hpp"
#include "../game/GameRules.hpp"
#include "Dealer.hpp"
#include "Shoe.hpp"
#include <array>
#include <memory>
#include <unordered_map>

namespace blackjack {
namespace solver {

/** The rules the solver models. Only what changes an EV is here. */
struct Rules {
  int decks = 6;
  bool hitsSoft17 = false;
  bool doubleAfterSplit = false;
  bool surrender = false; ///< Late surrender, after the peek
  double blackjackPayout = 1.5;
  /// The engine splits K,Q? No: Hand::canSplit compares ranks, so only a pair
  /// of identical ten ranks splits. Farmer's CA splits any two tens. The
  /// difference only matters to a policy that splits tens.
  bool splitIdenticalRanksOnly = true;

  /** The rules BlackjackGame actually plays (no DAS, identical-rank splits). */
  static Rules fromGame(const GameRules &game) {
    Rules r;
    r.decks = static_cast<int>(game.numDecks);
    r.hitsSoft17 = game.dealerHitsSoft17;
    r.doubleAfterSplit = false;
    r.surrender = game.surrender;
    r.blackjackPayout = game.blackjackPayout;
    r.splitIdenticalRanksOnly = true;
    return r;
  }
};

/** What a policy can see about a hand when it decides. */
struct HandInfo {
  const Composition &hand;
  int upCard;
  bool canDouble;
  bool canSplit;
  bool canSurrender;
  bool splitHand; ///< One of the two hands of a split
};

/** EV of each action from one decision point; NaN marks an illegal action. */
struct ActionValues {
  std::array<double, 5> v; ///< Indexed by ai::Action

  ActionValues();
  double operator[](ai::Action a) const { return v[static_cast<size_t>(a)]; }
  double &operator[](ai::Action a) { return v[static_cast<size_t>(a)]; }
  bool legal(ai::Action a) const;
  ai::Action best() const; ///< Ties go to the first in enum order
  double max() const { return (*this)[best()]; }
};

/** Chooses actions. Must be deterministic in its inputs. */
class Policy {
public:
  virtual ~Policy() = default;
  /**
   * @param unsplit The EV of each legal action for this same composition, as
   *   an unsplit hand. Only passed when needsUnsplitValues() is true; a
   *   policy that reads a chart has no use for it.
   */
  virtual ai::Action choose(const HandInfo &info,
                            const ActionValues *unsplit) const = 0;
  virtual bool needsUnsplitValues() const { return false; }
};

/** Composition-dependent optimal play (CDZ-). */
class OptimalPolicy : public Policy {
public:
  ai::Action choose(const HandInfo &info, const ActionValues *unsplit) const override;
  bool needsUnsplitValues() const override { return true; }
};

/**
 * Plays an ai::Agent greedily on exactly the State and legal actions the
 * engine would give it. The agent must be count-blind or have countBucket
 * fixed by the caller: the solver has one shoe, so one count.
 */
class AgentPolicy : public Policy {
public:
  explicit AgentPolicy(ai::Agent &agent, int countBucket = ai::COUNT_UNTRACKED)
      : agent_(agent), countBucket_(countBucket) {}
  ai::Action choose(const HandInfo &info, const ActionValues *unsplit) const override;

  /** The State BlackjackGame + GameStateConverter would build for this hand. */
  static ai::State stateFor(const HandInfo &info, int countBucket);

private:
  ai::Agent &agent_;
  int countBucket_;
};

class Solver {
public:
  /** Solve a full shoe of rules.decks decks. */
  Solver(const Rules &rules, const Policy &policy);
  /** Solve a depleted shoe, e.g. what is left partway through a dealt one. */
  Solver(const Rules &rules, const Shoe &shoe, const Policy &policy);
  ~Solver();
  Solver(const Solver &) = delete;
  Solver &operator=(const Solver &) = delete;

  /** EV per round of the policy, off the top of this shoe. */
  double expectedValue();
  /** Same, given the dealer shows `upCard` (still including the peek). */
  double expectedValue(int upCard);

  /** EV of every legal action on an opening two-card hand. */
  ActionValues openingValues(int card1, int card2, int upCard);
  /**
   * The policy's play on that hand. `canSplit` says whether a pair may split;
   * it is false for K,Q-style ten pairs under splitIdenticalRanksOnly, and
   * ignored for cards that differ.
   */
  ai::Action openingAction(int card1, int card2, int upCard, bool canSplit = true);
  /** P(the hand is a splittable pair | two ten-valued cards), see the rule. */
  double pairOfTensSameRank() const;
  /** EV of every legal action on any unsplit hand (2+ cards). */
  ActionValues handValues(const Composition &hand, int upCard);
  /** EV of splitting a pair of `pairCard` against `upCard`. */
  double splitValue(int pairCard, int upCard);
  /**
   * The values a player actually faces at the table: those of
   * openingValues() (two cards) or handValues() (more), given the dealer has
   * peeked and has no natural. These are the numbers Farmer's CA prints per
   * hand; v becomes (v + 1 - p) / p with p = P(no natural), a natural's own
   * value just v / p.
   */
  ActionValues peekedValues(const Composition &hand, int upCard);

  /** Dealer outcomes for this shoe, by upcard (nothing removed but the upcard). */
  std::array<DealerDist, 11> dealerOutcomes() const;

  const Rules &rules() const { return rules_; }
  const Shoe &shoe() const { return shoe_; }

private:
  struct Node;
  struct Context;

  Rules rules_;
  Shoe shoe_;
  const Policy &policy_;
  std::unique_ptr<Context> main_;
  std::array<std::unique_ptr<Context>, 11> split_; ///< One per pair card

  /// Dealer distributions keyed by every card removed from shoe_ (player
  /// cards plus a removed pair card), shared by all contexts.
  std::unordered_map<uint64_t, std::array<DealerDist, 11>> dealerCache_;

  const std::array<DealerDist, 11> &dealerFor(uint64_t removedKey, const Shoe &s);
  Node &node(Context &ctx, Composition &hand);
  Context &splitContext(int pairCard);
  double probabilityHoleNatural(const Shoe &afterUpCard, int upCard) const;
};

} // namespace solver
} // namespace blackjack