#include "PolicyTable.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace blackjack {
namespace ai {
namespace {

// A finished training run is expensive; losing it to a missing directory is
// not an acceptable failure. The build used to create these paths in the
// source tree at configure time, so saving only worked when the binary
// happened to be run from there.
void ensureParentDirectory(const std::string &filepath) {
  std::filesystem::path parent = std::filesystem::path(filepath).parent_path();
  if (parent.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
}

} // namespace

void PolicyTable::saveToBinary(const std::string &filepath) const {
  ensureParentDirectory(filepath);

  std::ofstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing: " + filepath);
  }

  // v2 adds per-action visit counts after each Q row. v1 files remain loadable.
  uint32_t version = 2;
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
    file.write(reinterpret_cast<const char *>(visits_[i].data()),
               sizeof(uint32_t) * NUM_ACTIONS);
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

  if (version != 1 && version != 2) {
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

    VisitCounts counts{};
    if (version >= 2) {
      file.read(reinterpret_cast<char *>(counts.data()),
                sizeof(uint32_t) * NUM_ACTIONS);
    } else {
      // v1 predates visit tracking. Mark every action of a stored state as
      // seen once: it is wrong in detail but keeps old checkpoints usable
      // without silently reporting every action as untrained.
      counts.fill(1);
    }

    size_t idx = state.hash();
    table_[idx] = qvalues;
    visits_[idx] = counts;
    visited_[idx] = true;
  }

  file.close();
}

void PolicyTable::exportToCSV(const std::string &filepath) const {
  ensureParentDirectory(filepath);

  std::ofstream file(filepath);
  if (!file) {
    throw std::runtime_error("Cannot open file for writing: " + filepath);
  }

  file << "player_total,dealer_card,usable_ace,can_split,can_double,"
          "Q_HIT,Q_STAND,Q_DOUBLE,Q_SPLIT,Q_SURRENDER,"
          "N_HIT,N_STAND,N_DOUBLE,N_SPLIT,N_SURRENDER\n";
  file << std::fixed << std::setprecision(6);

  for (size_t i = 0; i < TABLE_SIZE; ++i) {
    if (!visited_[i]) continue;

    State state = stateFromHash(i);

    // canSplit/canDouble are part of the hash, so leaving them out of the key
    // emits two rows that look identical but hold different policies.
    file << state.playerTotal << "," << state.dealerUpCard << ","
         << (state.hasUsableAce ? "1" : "0") << ","
         << (state.canSplit ? "1" : "0") << ","
         << (state.canDouble ? "1" : "0") << ",";

    for (size_t j = 0; j < NUM_ACTIONS; ++j) {
      file << table_[i][j] << ",";
    }

    for (size_t j = 0; j < NUM_ACTIONS; ++j) {
      file << visits_[i][j];
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
