#include "util/ArgParser.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace blackjack::util;

namespace {

/// ArgParser::parse takes (int, char**), so build a mutable argv from literals.
class Args {
public:
  explicit Args(std::vector<std::string> tokens) : storage_(std::move(tokens)) {
    for (auto &t : storage_) {
      argv_.push_back(t.data());
    }
  }

  int argc() const { return static_cast<int>(argv_.size()); }
  char **argv() { return argv_.data(); }

private:
  std::vector<std::string> storage_;
  std::vector<char *> argv_;
};

ArgParser makeParser() {
  ArgParser args("test", "test parser");
  args.addFlag("episodes", "e", "Episode count", "1000000");
  args.addFlag("checkpoint", "c", "Checkpoint path", "");
  args.addBool("verbose", "v", "Verbose output");
  args.addBool("help", "h", "Show help");
  return args;
}

} // namespace

// === has() vs wasProvided() ===

// A flag registered with a default satisfies has() even when absent from argv.
// train.cpp relied on has() to mean "user passed this", which silently let the
// built-in default override values loaded from a config file.
TEST(ArgParserTest, HasIsTrueForUnpassedFlagWithDefault) {
  ArgParser args = makeParser();
  Args cli({"test"});
  ASSERT_TRUE(args.parse(cli.argc(), cli.argv()));

  EXPECT_TRUE(args.has("episodes"));
  EXPECT_EQ(args.getString("episodes"), "1000000");
}

TEST(ArgParserTest, WasProvidedIsFalseForUnpassedFlagWithDefault) {
  ArgParser args = makeParser();
  Args cli({"test"});
  ASSERT_TRUE(args.parse(cli.argc(), cli.argv()));

  EXPECT_FALSE(args.wasProvided("episodes"));
}

TEST(ArgParserTest, WasProvidedIsTrueForPassedFlag) {
  ArgParser args = makeParser();
  Args cli({"test", "--episodes", "500"});
  ASSERT_TRUE(args.parse(cli.argc(), cli.argv()));

  EXPECT_TRUE(args.wasProvided("episodes"));
  EXPECT_EQ(args.getString("episodes"), "500");
}

TEST(ArgParserTest, WasProvidedTracksShortFormFlags) {
  ArgParser args = makeParser();
  Args cli({"test", "-e", "250"});
  ASSERT_TRUE(args.parse(cli.argc(), cli.argv()));

  EXPECT_TRUE(args.wasProvided("episodes"));
  EXPECT_EQ(args.getString("episodes"), "250");
}

TEST(ArgParserTest, WasProvidedTracksBoolFlags) {
  ArgParser args = makeParser();
  Args cli({"test", "--verbose"});
  ASSERT_TRUE(args.parse(cli.argc(), cli.argv()));

  EXPECT_TRUE(args.wasProvided("verbose"));
  EXPECT_FALSE(args.wasProvided("episodes"));
}

// Flags declared with an empty default are absent until passed, so has() and
// wasProvided() agree for them.
TEST(ArgParserTest, EmptyDefaultFlagIsAbsentUntilPassed) {
  ArgParser args = makeParser();
  Args cli({"test"});
  ASSERT_TRUE(args.parse(cli.argc(), cli.argv()));

  EXPECT_FALSE(args.has("checkpoint"));
  EXPECT_FALSE(args.wasProvided("checkpoint"));
}

// === Parse failures ===

TEST(ArgParserTest, UnknownOptionFailsParse) {
  ArgParser args = makeParser();
  Args cli({"test", "--nonsense"});

  EXPECT_FALSE(args.parse(cli.argc(), cli.argv()));
}

TEST(ArgParserTest, MissingValueFailsParse) {
  ArgParser args = makeParser();
  Args cli({"test", "--episodes"});

  EXPECT_FALSE(args.parse(cli.argc(), cli.argv()));
}

TEST(ArgParserTest, PositionalArgumentFailsParse) {
  ArgParser args = makeParser();
  Args cli({"test", "500000"});

  EXPECT_FALSE(args.parse(cli.argc(), cli.argv()));
}

// === Typed getters ===

TEST(ArgParserTest, GetIntParsesValue) {
  ArgParser args = makeParser();
  Args cli({"test", "--episodes", "12345"});
  ASSERT_TRUE(args.parse(cli.argc(), cli.argv()));

  EXPECT_EQ(args.getInt("episodes"), 12345);
}

TEST(ArgParserTest, GetStringOnMissingFlagThrows) {
  ArgParser args = makeParser();
  Args cli({"test"});
  ASSERT_TRUE(args.parse(cli.argc(), cli.argv()));

  EXPECT_THROW(args.getString("checkpoint"), std::runtime_error);
}
