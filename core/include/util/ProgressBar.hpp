#pragma once

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace blackjack {
namespace util {

class ProgressBar {
public:
  explicit ProgressBar(size_t total, size_t updateFrequency = 1000,
                       int barWidth = 40)
      : total_(total),
        updateFrequency_(updateFrequency),
        barWidth_(barWidth),
        silent_(false),
        startTime_(std::chrono::steady_clock::now()) {
#ifdef _WIN32
    silent_ = _isatty(_fileno(stdout)) == 0;
#else
    silent_ = !isatty(fileno(stdout));
#endif
  }

  void update(size_t current, const std::string& extraInfo = "") {
    if (silent_ || current % updateFrequency_ != 0) return;
    double pct = static_cast<double>(current) / total_;
    int filled = static_cast<int>(pct * barWidth_);

    std::string bar(static_cast<size_t>(filled), '=');
    if (filled < barWidth_) bar += '>';
    bar.resize(static_cast<size_t>(barWidth_), ' ');

    std::cout << "\r[" << bar << "] " << static_cast<int>(pct * 100) << "% | "
              << "Episode " << formatCount(current) << "/" << formatCount(total_);

    if (current > 0) {
      std::cout << " | ETA: " << formatETA(current);
    }
    if (!extraInfo.empty()) {
      std::cout << " | " << extraInfo;
    }
    std::cout << std::flush;
  }

  void finish(const std::string& finalInfo = "") {
    if (!silent_) {
      int filled = barWidth_;
      std::string bar(static_cast<size_t>(filled), '=');
      std::cout << "\r[" << bar << "] 100% | "
                << "Episode " << formatCount(total_) << "/" << formatCount(total_);
      if (total_ > 0) {
        std::cout << " | ETA: 0s";
      }
      if (!finalInfo.empty()) {
        std::cout << " | " << finalInfo;
      }
    }
    std::cout << "\n" << std::flush;
  }

  void setSilent(bool silent) { silent_ = silent; }

private:
  size_t total_;
  size_t updateFrequency_;
  int barWidth_;
  bool silent_;
  std::chrono::steady_clock::time_point startTime_;

  std::string formatCount(size_t n) const {
    if (n >= 1'000'000) return std::to_string(n / 1'000'000) + "M";
    if (n >= 1'000) return std::to_string(n / 1'000) + "K";
    return std::to_string(n);
  }

  std::string formatETA(size_t current) const {
    if (current == 0) return "?";
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - startTime_).count();
    if (elapsed <= 0) return "?";
    double rate = static_cast<double>(current) / elapsed;
    size_t remaining = static_cast<size_t>(
        (total_ - current) / rate);
    if (remaining >= 60) {
      return std::to_string(remaining / 60) + "m " +
             std::to_string(remaining % 60) + "s";
    }
    return std::to_string(remaining) + "s";
  }
};

}  // namespace util
}  // namespace blackjack
