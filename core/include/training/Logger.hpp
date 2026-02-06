#pragma once

#include <chrono>
#include <fstream>
#include <string>

namespace blackjack {
namespace training {

// Forward declaration - full definition in Trainer.hpp
struct TrainingMetrics;

/**
 * @brief Logs training progress to file
 *
 * Outputs CSV format for easy analysis in Python/Excel
 */
class Logger {
public:
  /**
   * @brief Construct logger with output directory
   */
  explicit Logger(const std::string &logDir);

  /**
   * @brief Destructor - closes log file
   */
  ~Logger();

  /**
   * @brief Log training metrics
   */
  void log(const TrainingMetrics &metrics);

  /**
   * @brief Flush log to disk
   */
  void flush();

  /**
   * @brief Get log file path
   */
  std::string getLogPath() const { return logPath_; }

private:
  std::string logDir_;
  std::string logPath_;
  std::ofstream logFile_;
  std::chrono::steady_clock::time_point startTime_;

  void writeHeader();
  std::string getCurrentTimestamp() const;
};
} // namespace training
} // namespace blackjack
