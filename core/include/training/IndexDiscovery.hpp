#pragma once

/**
 * @file IndexDiscovery.hpp
 * @brief Has a learned counter found the Illustrious 18?
 *
 * For each index play, read the agent's preference between the deviation
 * and the base play at every count bucket, and find every threshold that
 * explains those preferences with the fewest mismatches. The play counts as
 * rediscovered only if *all* of those thresholds lie within one true count
 * of the book's index. A preference read from a bucket the agent never
 * tried both plays in is no preference, so such buckets are skipped; an
 * agent that has only seen two buckets can't pin an index down, and this
 * says so instead of crediting it with the nearest number.
 *
 * Reads the agent through getQValue and getVisitCount only, so it scores any
 * count-aware agent the same way.
 */

#include "../ai/Agent.hpp"
#include "Evaluator.hpp"
#include <vector>

namespace blackjack {
namespace training {

struct LearnedIndex {
  IndexPlay play;
  int informedBuckets = 0; ///< Buckets where both plays have been tried
  int lowest = 0;          ///< Range of best-fitting thresholds; deviate when
  int highest = 0;         ///< floor(TC) >= threshold. 7 means never deviate.
  int mismatches = 0;      ///< Buckets the best threshold gets wrong
  bool rediscovered = false;
};

/** The agent's learned version of every index play in illustrious18(). */
std::vector<LearnedIndex> learnedIndices(const ai::Agent &agent);

/** How many of them it has rediscovered. */
int countRediscovered(const std::vector<LearnedIndex> &indices);

} // namespace training
} // namespace blackjack
