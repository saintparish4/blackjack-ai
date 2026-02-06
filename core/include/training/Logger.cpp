#include "Logger.hpp"
#include "Trainer.hpp"
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>


namespace blackjack {
namespace training {
Logger::Logger(const std::string &logDir)
    : logDir_(logDir), startTime_(std::chrono::steady_clock::now()) {
  // Create log directory
  std::filesystem::create_directories(logDir);

  // Generate log filename with timestamp
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#ifdef _WIN32
  localtime_s(&tm_buf, &time_t);
#else
  localtime_r(&time_t, &tm_buf);
#endif

  std::ostringstream oss;
  oss << logDir << "/training_"
      << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".csv";

  logPath_ = oss.str();

  // Open log file
  logFile_.open(logPath_, std::ios::out);

  if (!logFile_) {
    throw std::runtime_error("Cannot open log file: " + logPath_);
  }

  writeHeader();
}

Logger::~Logger() {
  if (logFile_.is_open()) {
    logFile_.close();
  }
}

void Logger::writeHeader() {
  logFile_ << "episode,elapsed_sec,win_rate,loss_rate,push_rate,"
           << "avg_reward,bust_rate,epsilon,states_learned\n";
  logFile_.flush();
}

void Logger::log(const TrainingMetrics &metrics) {
  auto now = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(now - startTime_)
          .count();

  logFile_ << metrics.totalEpisodes << "," << elapsed << "," << std::fixed
           << std::setprecision(6) << metrics.winRate << "," << metrics.lossRate
           << "," << metrics.pushRate << "," << metrics.avgReward << ","
           << metrics.bustRate << "," << metrics.currentEpsilon << ","
           << metrics.statesLearned << "\n";

  logFile_.flush();
}

void Logger::flush() { logFile_.flush(); }

std::string Logger::getCurrentTimestamp() const {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#ifdef _WIN32
  localtime_s(&tm_buf, &time_t);
#else
  localtime_r(&time_t, &tm_buf);
#endif

  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}
} // namespace training
} // namespace blackjack