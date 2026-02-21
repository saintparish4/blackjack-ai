#include "PolicyTable.hpp"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace blackjack {
namespace ai {

void PolicyTable::saveToBinary(const std::string &filepath) const {
  std::ofstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing: " + filepath);
  }

  uint32_t version = 1;
  uint64_t tableSize = visited_.count();

  file.write(reinterpret_cast<const char *>(&version), sizeof(version));
  file.write(reinterpret_cast<const char *>(&tableSize), sizeof(tableSize));

  for (size_t i = 0; i < TABLE_SIZE; ++i) {
    if (!visited_[i]) continue;

    State state = stateFromHash(i);

    file.write(reinterpret_cast<const char *>(&state.playerTotal),
               sizeof(state.playerTotal));
    file.write(reinterpret_cast<const char *>(&state.dealerUpCard),
               sizeof(state.dealerUpCard));
    file.write(reinterpret_cast<const char *>(&state.hasUsableAce),
               sizeof(state.hasUsableAce));
    file.write(reinterpret_cast<const char *>(&state.canSplit),
               sizeof(state.canSplit));
    file.write(reinterpret_cast<const char *>(&state.canDouble),
               sizeof(state.canDouble));

    file.write(reinterpret_cast<const char *>(table_[i].data()),
               sizeof(double) * NUM_ACTIONS);
  }

  file.close();
}

void PolicyTable::loadFromBinary(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file for reading: " + filepath);
  }

  uint32_t version;
  uint64_t tableSize;

  file.read(reinterpret_cast<char *>(&version), sizeof(version));
  file.read(reinterpret_cast<char *>(&tableSize), sizeof(tableSize));

  if (version != 1) {
    throw std::runtime_error("Unsupported file version");
  }

  clear();

  for (uint64_t i = 0; i < tableSize; ++i) {
    State state(0, 0, false, false, false);
    QValues qvalues;

    file.read(reinterpret_cast<char *>(&state.playerTotal),
              sizeof(state.playerTotal));
    file.read(reinterpret_cast<char *>(&state.dealerUpCard),
              sizeof(state.dealerUpCard));
    file.read(reinterpret_cast<char *>(&state.hasUsableAce),
              sizeof(state.hasUsableAce));
    file.read(reinterpret_cast<char *>(&state.canSplit),
              sizeof(state.canSplit));
    file.read(reinterpret_cast<char *>(&state.canDouble),
              sizeof(state.canDouble));

    file.read(reinterpret_cast<char *>(qvalues.data()),
              sizeof(double) * NUM_ACTIONS);

    size_t idx = state.hash();
    table_[idx] = qvalues;
    visited_[idx] = true;
  }

  file.close();
}

void PolicyTable::exportToCSV(const std::string &filepath) const {
  std::ofstream file(filepath);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing: " + filepath);
  }

  file << "player_total,dealer_card,usable_ace,Q_HIT,Q_STAND,Q_DOUBLE,Q_SPLIT,Q_SURRENDER\n";
  file << std::fixed << std::setprecision(6);

  for (size_t i = 0; i < TABLE_SIZE; ++i) {
    if (!visited_[i]) continue;

    State state = stateFromHash(i);

    file << state.playerTotal << "," << state.dealerUpCard << ","
         << (state.hasUsableAce ? "1" : "0") << ",";

    for (size_t j = 0; j < NUM_ACTIONS; ++j) {
      file << table_[i][j];
      if (j < NUM_ACTIONS - 1) {
        file << ",";
      }
    }

    file << "\n";
  }

  file.close();
}

} // namespace ai
} // namespace blackjack
