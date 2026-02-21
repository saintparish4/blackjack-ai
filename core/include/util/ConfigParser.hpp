#pragma once

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace blackjack {
namespace util {

/**
 * Simple INI-style config file parser.
 *
 * Format: key = value pairs, one per line.
 * '#' starts a comment (rest of line ignored).
 * Blank lines and comment-only lines are skipped.
 * Keys and values are whitespace-trimmed.
 *
 * Type coercion errors throw std::runtime_error with the offending key/value.
 * Missing keys return the caller-supplied default.
 */
class ConfigParser {
public:
    /** Load key-value pairs from a file. Throws if the file cannot be opened. */
    void load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + filepath);
        }
        std::string line;
        while (std::getline(file, line)) {
            auto commentPos = line.find('#');
            if (commentPos != std::string::npos) {
                line = line.substr(0, commentPos);
            }
            line = trim(line);
            if (line.empty()) continue;

            auto eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key   = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));
            if (!key.empty()) {
                values_[key] = value;
            }
        }
    }

    bool has(const std::string& key) const {
        return values_.count(key) > 0;
    }

    std::string getString(const std::string& key,
                          const std::string& defaultVal = "") const {
        auto it = values_.find(key);
        return (it != values_.end()) ? it->second : defaultVal;
    }

    int getInt(const std::string& key, int defaultVal = 0) const {
        auto it = values_.find(key);
        if (it == values_.end()) return defaultVal;
        try {
            return std::stoi(it->second);
        } catch (...) {
            throw std::runtime_error("Config key '" + key +
                "': expected int, got '" + it->second + "'");
        }
    }

    double getDouble(const std::string& key, double defaultVal = 0.0) const {
        auto it = values_.find(key);
        if (it == values_.end()) return defaultVal;
        try {
            return std::stod(it->second);
        } catch (...) {
            throw std::runtime_error("Config key '" + key +
                "': expected double, got '" + it->second + "'");
        }
    }

    /** Accepts: true/false, 1/0, yes/no (case-sensitive). */
    bool getBool(const std::string& key, bool defaultVal = false) const {
        auto it = values_.find(key);
        if (it == values_.end()) return defaultVal;
        const std::string& v = it->second;
        if (v == "true"  || v == "1" || v == "yes") return true;
        if (v == "false" || v == "0" || v == "no")  return false;
        throw std::runtime_error("Config key '" + key +
            "': expected bool (true/false/1/0/yes/no), got '" + v + "'");
    }

    /** Raw key-value map exposed for diagnostics and iteration. */
    const std::map<std::string, std::string>& all() const { return values_; }

private:
    std::map<std::string, std::string> values_;

    static std::string trim(const std::string& s) {
        const char* ws = " \t\r\n";
        size_t start = s.find_first_not_of(ws);
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(ws);
        return s.substr(start, end - start + 1);
    }
};

} // namespace util
} // namespace blackjack
