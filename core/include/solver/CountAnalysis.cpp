#include "CountAnalysis.hpp"

#include <cmath>

namespace blackjack {
namespace solver {

Tags hiLoTags() {
  Tags t{};
  for (int r = 2; r <= 6; ++r) t[static_cast<size_t>(r)] = 1.0;
  t[ACE] = -1.0;
  t[TEN] = -1.0;
  return t;
}

std::array<double, 11> effectOfRemoval(const Rules &rules, const Shoe &shoe,
                                       const Policy &policy) {
  Solver full(rules, shoe, policy);
  const double base = full.expectedValue();
  std::array<double, 11> e{};
  for (int r = 1; r <= 10; ++r) {
    Shoe less = shoe;
    less.remove(r);
    Solver s(rules, less, policy);
    e[static_cast<size_t>(r)] = s.expectedValue() - base;
  }
  return e;
}

namespace {

/// Weighted least squares of effects on tags, effects centred; returns
/// {slope, correlation}.
std::pair<double, double> regress(const std::array<double, 11> &e, const Tags &t,
                                  const Shoe &shoe) {
  double w = 0.0, mean = 0.0;
  for (int r = 1; r <= 10; ++r) {
    w += shoe.n[r];
    mean += shoe.n[r] * e[static_cast<size_t>(r)];
  }
  mean /= w;
  double et = 0.0, ee = 0.0, tt = 0.0;
  for (int r = 1; r <= 10; ++r) {
    const double c = e[static_cast<size_t>(r)] - mean;
    const double tag = t[static_cast<size_t>(r)];
    et += shoe.n[r] * c * tag;
    ee += shoe.n[r] * c * c;
    tt += shoe.n[r] * tag * tag;
  }
  const double slope = tt > 0 ? et / tt : 0.0;
  const double corr = ee > 0 && tt > 0 ? et / std::sqrt(ee * tt) : 0.0;
  return {slope, corr};
}

double gainOf(const Rules &rules, const Shoe &shoe, const Policy &policy,
              const Composition &hand, int up, ai::Action deviation, ai::Action base) {
  Solver s(rules, shoe, policy);
  const ActionValues v = s.peekedValues(hand, up);
  return v[deviation] - v[base];
}

} // namespace

double weightedCorrelation(const std::array<double, 11> &effects, const Tags &tags,
                           const Shoe &shoe) {
  return regress(effects, tags, shoe).second;
}

IndexEstimate estimateIndex(const Rules &rules, const Shoe &shoe, const Policy &policy,
                            const Composition &hand, int upCard, ai::Action deviation,
                            ai::Action base, const Tags &tags) {
  IndexEstimate est;
  est.gain = gainOf(rules, shoe, policy, hand, upCard, deviation, base);
  // The cards on the table are already out of the shoe the EoRs perturb.
  Shoe left = shoe;
  for (int r = 1; r <= 10; ++r) {
    left.n[r] -= hand.c[r];
    left.total -= hand.c[r];
  }
  left.remove(upCard);
  for (int r = 1; r <= 10; ++r) {
    if (left.n[r] == 0) continue;
    Shoe less = shoe;
    less.remove(r);
    est.eor[static_cast<size_t>(r)] =
        gainOf(rules, less, policy, hand, upCard, deviation, base) - est.gain;
  }
  const auto [slope, corr] = regress(est.eor, tags, left);
  est.slopePerTrueCount = slope * shoe.total / 52.0;
  est.correlation = corr;
  est.linearIndex = est.slopePerTrueCount != 0.0 ? -est.gain / est.slopePerTrueCount : 0.0;
  return est;
}

} // namespace solver
} // namespace blackjack
