#include "State.hpp"
#include <sstream>

namespace blackjack {
namespace ai {

std::string State::toString() const {
  std::ostringstream oss;
  oss << "State(player=" << playerTotal;

  if (hasUsableAce) {
    oss << " soft";
  }

  oss << ", dealer=" << dealerUpCard;

  if (canSplit) {
    oss << ", canSplit";
  }

  if (canDouble) {
    oss << ", canDouble";
  }

  oss << ")";
  return oss.str();
}
} // namespace ai
} // namespace blackjack