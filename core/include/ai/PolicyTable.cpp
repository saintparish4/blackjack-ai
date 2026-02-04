#include "PolicyTable.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace blackjack {
namespace ai {
void PolicyTable::saveToBinary(const std::string &filepath) const {
  std::ofstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing: " + filepath);
  }

  // Write header: version and size
  uint32_t version = 1;
  uint64_t tableSize = table_.size();

  file.write(reinterpret_cast<const char *>(&version), sizeof(version));
  file.write(reinterpret_cast<const char *>(&tableSize), sizeof(tableSize));

  // Write each state-qvalues pair
  for (const auto &[state, qvalues] : table_) {
    // Write state
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

    // Write Q-values
    file.write(reinterpret_cast<const char *>(qvalues.data()),
               sizeof(double) * NUM_ACTIONS);
  }

  file.close();
}

void PolicyTable::loadFromBinary(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file for reading: " + filepath);
  }

  // Read header
  uint32_t version;
  uint64_t tableSize;

  file.read(reinterpret_cast<char *>(&version), sizeof(version));
  file.read(reinterpret_cast<char *>(&tableSize), sizeof(tableSize));

  if (version != 1) {
    throw std::runtime_error("Unsupported file version");
  }

  // Clear existing table
  table_.clear();

  // Read each entry
  for (uint64_t i = 0; i < tableSize; ++i) {
    State state(0, 0, false, false, false);
    QValues qvalues;

    // Read state
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

    // Read Q-values
    file.read(reinterpret_cast<char *>(qvalues.data()),
              sizeof(double) * NUM_ACTIONS);

    table_[state] = qvalues;
  }

  file.close();
}

void PolicyTable::exportToCSV(const std::string &filepath) const {
  std::ofstream file(filepath);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing: " + filepath);
  }

  // Write header
  file
      << "player_total,dealer_card,usable_ace,Q_HIT,Q_STAND,Q_DOUBLE,Q_SPLIT\n";

  // Write data
  file << std::fixed << std::setprecision(6);

  for (const auto &[state, qvalues] : table_) {
    file << state.playerTotal << "," << state.dealerUpCard << ","
         << (state.hasUsableAce ? "1" : "0") << ",";

    for (size_t i = 0; i < NUM_ACTIONS; ++i) {
      file << qvalues[i];
      if (i < NUM_ACTIONS - 1) {
        file << ",";
      }
    }

    file << "\n";
  }

  file.close();
}

} // namespace ai
} // namespace blackjack