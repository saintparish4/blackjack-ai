#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <stdexcept>

namespace blackjack {
namespace util {

class ArgParser {
public:
  ArgParser(const std::string& programName, const std::string& description)
      : programName_(programName), description_(description) {}

  void addFlag(const std::string& longName, const std::string& shortName,
               const std::string& description,
               const std::string& defaultValue = "", bool required = false) {
    flagDefs_.push_back({longName, shortName, description, defaultValue, false, required});
    if (!shortName.empty()) shortToLong_[shortName] = longName;
    if (!defaultValue.empty()) values_[longName] = defaultValue;
  }

  void addBool(const std::string& longName, const std::string& shortName,
               const std::string& description) {
    flagDefs_.push_back({longName, shortName, description, "", true, false});
    if (!shortName.empty()) shortToLong_[shortName] = longName;
  }

  bool parse(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      std::string key;

      if (arg.rfind("--", 0) == 0) {
        key = arg.substr(2);
      } else if (arg.rfind("-", 0) == 0 && arg.size() > 1) {
        std::string shortKey = arg.substr(1);
        auto it = shortToLong_.find(shortKey);
        if (it == shortToLong_.end()) {
          std::cerr << "Unknown option: " << arg << "\n";
          printHelp(std::cerr);
          return false;
        }
        key = it->second;
      } else {
        std::cerr << "Unexpected argument: " << arg << "\n";
        printHelp(std::cerr);
        return false;
      }

      const FlagDef* def = findDef(key);
      if (!def) {
        std::cerr << "Unknown option: --" << key << "\n";
        printHelp(std::cerr);
        return false;
      }

      if (def->isBool) {
        values_[key] = "true";
      } else {
        if (i + 1 >= argc) {
          std::cerr << "Option --" << key << " requires a value.\n";
          return false;
        }
        values_[key] = argv[++i];
      }
    }

    if (has("help")) {
      printHelp();
      return false;
    }

    for (const auto& def : flagDefs_) {
      if (def.required && values_.find(def.longName) == values_.end()) {
        std::cerr << "Missing required option: --" << def.longName << "\n";
        printHelp(std::cerr);
        return false;
      }
    }
    return true;
  }

  bool has(const std::string& longName) const {
    return values_.find(longName) != values_.end();
  }

  std::string getString(const std::string& longName) const {
    auto it = values_.find(longName);
    if (it == values_.end())
      throw std::runtime_error("ArgParser: no value for --" + longName);
    return it->second;
  }

  int getInt(const std::string& longName) const {
    return std::stoi(getString(longName));
  }

  double getDouble(const std::string& longName) const {
    return std::stod(getString(longName));
  }

  bool getBool(const std::string& longName) const {
    if (!has(longName)) return false;
    std::string v = getString(longName);
    return v == "true" || v == "1" || v == "yes";
  }

  void printHelp(std::ostream& out = std::cout) const {
    out << "Usage: " << programName_ << " [options]\n";
    out << description_ << "\n\nOptions:\n";
    for (const auto& def : flagDefs_) {
      out << "  --" << def.longName;
      if (!def.shortName.empty()) out << ", -" << def.shortName;
      out << "\t" << def.description;
      if (!def.defaultValue.empty()) out << " (default: " << def.defaultValue << ")";
      if (def.required) out << " [required]";
      out << "\n";
    }
  }

private:
  struct FlagDef {
    std::string longName;
    std::string shortName;
    std::string description;
    std::string defaultValue;
    bool isBool;
    bool required;
  };

  const FlagDef* findDef(const std::string& longName) const {
    for (const auto& def : flagDefs_) {
      if (def.longName == longName) return &def;
    }
    return nullptr;
  }

  std::string programName_;
  std::string description_;
  std::vector<FlagDef> flagDefs_;
  std::map<std::string, std::string> values_;
  std::map<std::string, std::string> shortToLong_;
};

} // namespace util
} // namespace blackjack
