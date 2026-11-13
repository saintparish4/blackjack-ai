#include "IndexDiscovery.hpp"

#include <limits>

namespace blackjack {
namespace training {

std::vector<LearnedIndex> learnedIndices(const ai::Agent &agent) {
  std::vector<LearnedIndex> out;
  const int firstBucket = ai::countBucketFor(ai::MIN_TRUE_COUNT);
  const int lastBucket = ai::countBucketFor(ai::MAX_TRUE_COUNT);

  for (const IndexPlay &play : illustrious18()) {
    LearnedIndex li;
    li.play = play;
    // Which buckets prefer the deviation, among those that tried both plays.
    std::vector<std::pair<int, bool>> seen; // (floor of TC, deviates)
    for (int b = firstBucket; b <= lastBucket; ++b) {
      // The opening hand: two cards, so doubling is open; a pair if T,T.
      ai::State s(play.total, play.dealer, false, play.pair, true);
      s.countBucket = b;
      if (agent.getVisitCount(s, play.atOrAbove) == 0 ||
          agent.getVisitCount(s, play.below) == 0) {
        continue;
      }
      seen.emplace_back(ai::trueCountOfBucket(b),
                        agent.getQValue(s, play.atOrAbove) > agent.getQValue(s, play.below));
    }
    li.informedBuckets = static_cast<int>(seen.size());

    int best = std::numeric_limits<int>::max();
    for (int t = ai::MIN_TRUE_COUNT; t <= ai::MAX_TRUE_COUNT + 1; ++t) {
      int wrong = 0;
      for (const auto &[tc, deviates] : seen) wrong += deviates != (tc >= t) ? 1 : 0;
      if (wrong < best) {
        best = wrong;
        li.lowest = li.highest = t;
      } else if (wrong == best) {
        li.highest = t;
      }
    }
    li.mismatches = best;
    li.rediscovered = !seen.empty() && li.lowest >= play.index - 1 &&
                      li.highest <= play.index + 1;
    out.push_back(li);
  }
  return out;
}

int countRediscovered(const std::vector<LearnedIndex> &indices) {
  int n = 0;
  for (const LearnedIndex &li : indices) n += li.rediscovered ? 1 : 0;
  return n;
}

} // namespace training
} // namespace blackjack
