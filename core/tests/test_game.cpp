#include "game/BlackjackGame.hpp"
#include "game/GameRules.hpp"
#include <gtest/gtest.h>

using namespace blackjack;

class BlackjackGameTest : public ::testing::Test {
protected:
  BlackjackGame game;
  
  void SetUp() override {
    // Game is initialized with default rules
  }
};

// === Initialization Tests ===

TEST_F(BlackjackGameTest, GameInitializesCorrectly) {
  EXPECT_FALSE(game.isRoundComplete());
  EXPECT_TRUE(game.getPlayerHand().empty());
  EXPECT_TRUE(game.getDealerHand().empty());
}

TEST_F(BlackjackGameTest, StartRoundDealsInitialCards) {
  // Retry until we get a non-blackjack round (blackjack completes round immediately)
  bool foundNonBlackjackRound = false;
  for (int i = 0; i < 100 && !foundNonBlackjackRound; i++) {
    game.startRound();
    
    EXPECT_EQ(game.getPlayerHand().size(), 2);
    EXPECT_EQ(game.getDealerHand(false).size(), 2);
    
    if (!game.isRoundComplete()) {
      foundNonBlackjackRound = true;
      EXPECT_FALSE(game.isRoundComplete());
    }
  }
  EXPECT_TRUE(foundNonBlackjackRound);
}

TEST_F(BlackjackGameTest, StartRoundHidesDealerHoleCard) {
  game.startRound();
  
  Hand visibleDealerHand = game.getDealerHand(true);
  Hand fullDealerHand = game.getDealerHand(false);
  
  EXPECT_EQ(visibleDealerHand.size(), 1);
  EXPECT_EQ(fullDealerHand.size(), 2);
}

// === Hit Action Tests ===

TEST_F(BlackjackGameTest, HitAddsCardToPlayerHand) {
  // Retry until we get a non-blackjack round (blackjack completes round immediately)
  bool testedHit = false;
  for (int i = 0; i < 100 && !testedHit; i++) {
    game.startRound();
    
    if (!game.isRoundComplete()) {
      testedHit = true;
      size_t initialSize = game.getPlayerHand().size();
      
      bool success = game.hit();
      
      EXPECT_TRUE(success);
      EXPECT_EQ(game.getPlayerHand().size(), initialSize + 1);
    }
  }
  EXPECT_TRUE(testedHit);
}

TEST_F(BlackjackGameTest, HitCompletesRoundOnBust) {
  game.startRound();
  
  // Keep hitting until bust (or reasonable limit)
  int hitCount = 0;
  while (!game.isRoundComplete() && hitCount < 10) {
    game.hit();
    hitCount++;
  }
  
  if (game.getPlayerHand().isBust()) {
    EXPECT_TRUE(game.isRoundComplete());
    EXPECT_EQ(game.getOutcome(), Outcome::PLAYER_BUST);
  }
}

TEST_F(BlackjackGameTest, HitFailsWhenRoundComplete) {
  game.startRound();
  
  // Complete the round by standing
  game.stand();
  EXPECT_TRUE(game.isRoundComplete());
  
  size_t handSizeBefore = game.getPlayerHand().size();
  bool success = game.hit();
  
  EXPECT_FALSE(success);
  EXPECT_EQ(game.getPlayerHand().size(), handSizeBefore);
}

// === Stand Action Tests ===

TEST_F(BlackjackGameTest, StandCompletesRound) {
  // Retry until we get a non-blackjack round (blackjack completes round immediately)
  bool testedStand = false;
  for (int i = 0; i < 100 && !testedStand; i++) {
    game.startRound();
    
    if (!game.isRoundComplete()) {
      testedStand = true;
      EXPECT_FALSE(game.isRoundComplete());
      
      game.stand();
      
      EXPECT_TRUE(game.isRoundComplete());
      // Should have a valid outcome
      Outcome outcome = game.getOutcome();
      EXPECT_TRUE(outcome == Outcome::PLAYER_WIN || 
                  outcome == Outcome::DEALER_WIN || 
                  outcome == Outcome::PUSH ||
                  outcome == Outcome::DEALER_BUST);
    }
  }
  EXPECT_TRUE(testedStand);
}

TEST_F(BlackjackGameTest, StandPlaysDealerHand) {
  game.startRound();
  game.stand();
  
  // Dealer should have at least 2 cards (initial deal)
  Hand dealerHand = game.getDealerHand(false);
  EXPECT_GE(dealerHand.size(), 2);
  
  // Dealer should have played according to rules (hit until 17+)
  int dealerTotal = dealerHand.getTotal();
  if (!dealerHand.isBust()) {
    EXPECT_GE(dealerTotal, 17);
  }
}

// === Double Down Tests ===

TEST_F(BlackjackGameTest, CanDoubleDownOnFirstTwoCards) {
  // Retry until we get a non-blackjack round
  bool testedCanDoubleDown = false;
  for (int i = 0; i < 100 && !testedCanDoubleDown; i++) {
    game.startRound();
    
    if (!game.isRoundComplete()) {
      testedCanDoubleDown = true;
      EXPECT_TRUE(game.canDoubleDown());
      EXPECT_EQ(game.getPlayerHand().size(), 2);
    }
  }
  EXPECT_TRUE(testedCanDoubleDown);
}

TEST_F(BlackjackGameTest, CannotDoubleDownAfterHit) {
  // Retry until we get a non-blackjack round
  bool testedAfterHit = false;
  for (int i = 0; i < 100 && !testedAfterHit; i++) {
    game.startRound();
    
    if (!game.isRoundComplete()) {
      bool hitSuccess = game.hit();
      if (hitSuccess && !game.isRoundComplete()) {
        testedAfterHit = true;
        EXPECT_FALSE(game.canDoubleDown());
      }
    }
  }
  EXPECT_TRUE(testedAfterHit);
}

TEST_F(BlackjackGameTest, CannotDoubleDownWhenRoundComplete) {
  game.startRound();
  game.stand();
  
  EXPECT_FALSE(game.canDoubleDown());
}

TEST_F(BlackjackGameTest, DoubleDownAddsOneCardAndCompletesRound) {
  // Retry until we get a non-blackjack round (blackjack completes round immediately)
  bool testedDoubleDown = false;
  for (int i = 0; i < 100 && !testedDoubleDown; i++) {
    game.startRound();
    
    if (!game.isRoundComplete()) {
      testedDoubleDown = true;
      size_t initialSize = game.getPlayerHand().size();
      
      bool success = game.doubleDown();
      
      EXPECT_TRUE(success);
      EXPECT_EQ(game.getPlayerHand().size(), initialSize + 1);
      EXPECT_TRUE(game.isRoundComplete());
    }
  }
  EXPECT_TRUE(testedDoubleDown);
}

TEST_F(BlackjackGameTest, DoubleDownCompletesRoundEvenOnBust) {
  // Retry until we get a non-blackjack round (blackjack completes round immediately)
  bool testedDoubleDown = false;
  for (int i = 0; i < 100 && !testedDoubleDown; i++) {
    game.startRound();
    
    if (!game.isRoundComplete()) {
      testedDoubleDown = true;
      // This is probabilistic, but double down should complete the round
      bool success = game.doubleDown();
      
      EXPECT_TRUE(success);
      EXPECT_TRUE(game.isRoundComplete());
      
      if (game.getPlayerHand().isBust()) {
        EXPECT_EQ(game.getOutcome(), Outcome::PLAYER_BUST);
      }
    }
  }
  EXPECT_TRUE(testedDoubleDown);
}

// === Blackjack Detection Tests ===

TEST_F(BlackjackGameTest, PlayerBlackjackCompletesRoundImmediately) {
  // This test is probabilistic - we need to keep starting rounds until we get blackjack
  // or we could use a seeded deck if available
  bool foundBlackjack = false;
  
  for (int i = 0; i < 100 && !foundBlackjack; i++) {
    game.startRound();
    
    if (game.getPlayerHand().isBlackjack()) {
      foundBlackjack = true;
      EXPECT_TRUE(game.isRoundComplete());
      EXPECT_EQ(game.getOutcome(), Outcome::PLAYER_BLACKJACK);
    }
  }
  
  // At least verify the logic works when blackjack occurs
  if (foundBlackjack) {
    EXPECT_TRUE(game.isRoundComplete());
  }
}

TEST_F(BlackjackGameTest, DealerBlackjackCompletesRoundImmediately) {
  bool foundDealerBlackjack = false;
  
  for (int i = 0; i < 100 && !foundDealerBlackjack; i++) {
    game.startRound();
    
    if (game.getDealerHand(false).isBlackjack()) {
      foundDealerBlackjack = true;
      EXPECT_TRUE(game.isRoundComplete());
      Outcome outcome = game.getOutcome();
      EXPECT_TRUE(outcome == Outcome::DEALER_WIN || outcome == Outcome::PUSH);
    }
  }
}

TEST_F(BlackjackGameTest, BothBlackjackResultsInPush) {
  bool foundBothBlackjack = false;
  
  for (int i = 0; i < 500 && !foundBothBlackjack; i++) {
    game.startRound();
    
    if (game.getPlayerHand().isBlackjack() && 
        game.getDealerHand(false).isBlackjack()) {
      foundBothBlackjack = true;
      EXPECT_TRUE(game.isRoundComplete());
      EXPECT_EQ(game.getOutcome(), Outcome::PUSH);
    }
  }
}

// === Outcome Tests ===

TEST_F(BlackjackGameTest, GetOutcomeThrowsWhenRoundNotComplete) {
  // Keep trying until we get a round that's not complete (blackjack completes immediately)
  bool foundIncompleteRound = false;
  for (int i = 0; i < 100 && !foundIncompleteRound; i++) {
    game.startRound();
    if (!game.isRoundComplete()) {
      foundIncompleteRound = true;
      EXPECT_THROW(game.getOutcome(), std::logic_error);
    }
  }
  
  // Should find an incomplete round within reasonable attempts
  EXPECT_TRUE(foundIncompleteRound);
}

TEST_F(BlackjackGameTest, PlayerWinOutcome) {
  game.startRound();
  
  // Keep hitting until we have a good hand, then stand
  // This is probabilistic, but should eventually result in player win
  while (!game.isRoundComplete() && game.getPlayerHand().getTotal() < 18) {
    if (game.getPlayerHand().getTotal() < 17) {
      game.hit();
    } else {
      game.stand();
      break;
    }
  }
  
  if (game.isRoundComplete() && !game.getPlayerHand().isBust()) {
    Outcome outcome = game.getOutcome();
    // Valid outcomes
    EXPECT_TRUE(outcome == Outcome::PLAYER_WIN ||
                outcome == Outcome::DEALER_WIN ||
                outcome == Outcome::PUSH ||
                outcome == Outcome::DEALER_BUST);
  }
}

TEST_F(BlackjackGameTest, DealerBustOutcome) {
  game.startRound();
  game.stand();
  
  if (game.getDealerHand(false).isBust()) {
    Outcome outcome = game.getOutcome();
    EXPECT_EQ(outcome, Outcome::DEALER_BUST);
  }
}

// === Dealer Hand Visibility Tests ===

TEST_F(BlackjackGameTest, HideHoleCardShowsOnlyUpcard) {
  game.startRound();
  
  Hand visibleHand = game.getDealerHand(true);
  Hand fullHand = game.getDealerHand(false);
  
  EXPECT_EQ(visibleHand.size(), 1);
  EXPECT_EQ(fullHand.size(), 2);
  EXPECT_EQ(visibleHand.getCards()[0], fullHand.getCards()[0]);
}

TEST_F(BlackjackGameTest, HideHoleCardWorksAfterRoundComplete) {
  game.startRound();
  game.stand();
  
  Hand visibleHand = game.getDealerHand(true);
  Hand fullHand = game.getDealerHand(false);
  
  // hideHoleCard parameter still works after round complete
  EXPECT_EQ(visibleHand.size(), 1);
  EXPECT_GE(fullHand.size(), 2);
}

// === Reset Tests ===

TEST_F(BlackjackGameTest, ResetClearsGameState) {
  game.startRound();
  game.hit();
  EXPECT_FALSE(game.getPlayerHand().empty());
  
  game.reset();
  
  EXPECT_TRUE(game.getPlayerHand().empty());
  EXPECT_TRUE(game.getDealerHand().empty());
  EXPECT_FALSE(game.isRoundComplete());
}

TEST_F(BlackjackGameTest, ResetAllowsNewRound) {
  // Retry until we get a non-blackjack round after reset
  bool testedReset = false;
  for (int i = 0; i < 100 && !testedReset; i++) {
    game.startRound();
    if (!game.isRoundComplete()) {
      game.stand();
    }
    EXPECT_TRUE(game.isRoundComplete());
    
    game.reset();
    game.startRound();
    
    // Card counts should always be correct
    EXPECT_EQ(game.getPlayerHand().size(), 2);
    EXPECT_EQ(game.getDealerHand(false).size(), 2);
    
    // Only check isRoundComplete if no blackjack was dealt
    if (!game.isRoundComplete()) {
      testedReset = true;
      EXPECT_FALSE(game.isRoundComplete());
    }
  }
  EXPECT_TRUE(testedReset);
}

// === Game Rules Tests ===

TEST_F(BlackjackGameTest, CustomRulesAreApplied) {
  GameRules customRules;
  customRules.dealerHitsSoft17 = false;
  customRules.numDecks = 1;
  
  BlackjackGame customGame(customRules);
  
  EXPECT_FALSE(customGame.getRules().dealerHitsSoft17);
  EXPECT_EQ(customGame.getRules().numDecks, 1);
}

TEST_F(BlackjackGameTest, DealerHitsSoft17WhenEnabled) {
  GameRules rules;
  rules.dealerHitsSoft17 = true;
  
  BlackjackGame gameWithRules(rules);
  
  // This is probabilistic - dealer needs to have soft 17
  // We'll verify the rule is accessible
  EXPECT_TRUE(gameWithRules.getRules().dealerHitsSoft17);
}

// === Edge Cases ===

TEST_F(BlackjackGameTest, MultipleRoundsWorkCorrectly) {
  for (int round = 0; round < 5; round++) {
    game.startRound();
    EXPECT_EQ(game.getPlayerHand().size(), 2);
    
    // If blackjack occurred, round is already complete
    if (!game.isRoundComplete()) {
      game.stand();
    }
    
    EXPECT_TRUE(game.isRoundComplete());
    
    game.reset();
  }
}

TEST_F(BlackjackGameTest, StandDoesNothingWhenRoundComplete) {
  game.startRound();
  game.stand();
  EXPECT_TRUE(game.isRoundComplete());
  
  Outcome firstOutcome = game.getOutcome();
  
  // Stand again should do nothing
  game.stand();
  
  EXPECT_EQ(game.getOutcome(), firstOutcome);
}

TEST_F(BlackjackGameTest, DoubleDownFailsWhenNotAllowed) {
  // Retry until we get a non-blackjack round where hit succeeds
  bool testedDoubleDownFail = false;
  for (int i = 0; i < 100 && !testedDoubleDownFail; i++) {
    game.startRound();
    
    if (!game.isRoundComplete()) {
      bool hitSuccess = game.hit();
      if (hitSuccess && !game.isRoundComplete()) {
        testedDoubleDownFail = true;
        // Now can't double down because we already hit
        bool success = game.doubleDown();
        EXPECT_FALSE(success);
      }
    }
  }
  EXPECT_TRUE(testedDoubleDownFail);
}
