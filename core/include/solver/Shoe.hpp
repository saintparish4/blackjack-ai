#pragma once

/**
 * @file Shoe.hpp
 * @brief Card compositions for the exact solver: a shoe, and a player hand.
 *
 * The player hand is called Composition, not Hand, so that it never collides
 * with the engine's blackjack::Hand in a file that uses both namespaces.
 *
 * The solver works on value classes, not cards: 1 = ace, 2..9, 10 = any
 * ten-valued card. Suits never matter and the four ten ranks only matter for
 * one rule (this engine splits a pair only when the ranks are identical),
 * which Solver handles explicitly rather than by carrying ranks here.
 *
 * Both types are plain values with O(1) add/remove, because the solver's
 * recursions mutate a shoe in place and undo each change on the way back up.
 * Nothing here allocates.
 */

#include <array>
#include <cstdint>

namespace blackjack {
namespace solver {

/// Value classes 1 (ace) .. 10 (ten-valued). Index 0 is unused everywhere.
constexpr int ACE = 1;
constexpr int TEN = 10;

/** Cards left in a shoe, by value class. */
struct Shoe {
  std::array<int, 11> n{};
  int total = 0;

  /** A full shoe of `decks` 52-card decks: 4 of each class, 16 tens, per deck. */
  static Shoe decks(int decks) {
    Shoe s;
    for (int r = 1; r <= 9; ++r) s.n[r] = 4 * decks;
    s.n[TEN] = 16 * decks;
    s.total = 52 * decks;
    return s;
  }

  double p(int r) const { return static_cast<double>(n[r]) / total; }
  void remove(int r) { --n[r]; --total; }
  void add(int r) { ++n[r]; ++total; }
};

/**
 * A hand as a composition. key() identifies the composition regardless of the
 * order the cards arrived in, which is what makes the solver's memo tables
 * composition-dependent: two orderings of the same cards share one entry.
 */
struct Composition {
  std::array<uint8_t, 11> c{};
  int hardSum = 0; ///< Aces counted as 1
  int cards = 0;
  uint64_t packed = 0; ///< 5 bits per class; a hand holds at most 21 of any

  void add(int r) {
    ++c[r];
    hardSum += r;
    ++cards;
    packed += uint64_t{1} << (5 * (r - 1));
  }
  void remove(int r) {
    --c[r];
    hardSum -= r;
    --cards;
    packed -= uint64_t{1} << (5 * (r - 1));
  }

  bool soft() const { return c[ACE] > 0 && hardSum + 10 <= 21; }
  int total() const { return soft() ? hardSum + 10 : hardSum; }
  bool bust() const { return hardSum > 21; }
  uint64_t key() const { return packed; }
};

} // namespace solver
} // namespace blackjack