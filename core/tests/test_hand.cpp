#include "game/Card.hpp"
#include "game/Hand.hpp"
#include <gtest/gtest.h>


using namespace blackjack;

class HandTest : public ::testing::Test {
protected:
  Hand hand;
};

TEST_F(HandTest, EmptyHandHasZeroValue) {
  EXPECT_EQ(hand.getTotal(), 0);
  EXPECT_FALSE(hand.isSoft());
  EXPECT_TRUE(hand.empty());
}

TEST_F(HandTest, SingleCardValue) {
  hand.addCard(Card(Rank::FIVE, Suit::HEARTS));
  EXPECT_EQ(hand.getTotal(), 5);
  EXPECT_FALSE(hand.isSoft());
}

TEST_F(HandTest, FaceCardsWorthTen) {
  hand.addCard(Card(Rank::JACK, Suit::SPADES));
  EXPECT_EQ(hand.getTotal(), 10);

  hand.clear();
  hand.addCard(Card(Rank::QUEEN, Suit::HEARTS));
  EXPECT_EQ(hand.getTotal(), 10);

  hand.clear();
  hand.addCard(Card(Rank::KING, Suit::DIAMONDS));
  EXPECT_EQ(hand.getTotal(), 10);
}

TEST_F(HandTest, SoftAceHandling) {
  // Ace + 6 = 17 (soft)
  hand.addCard(Card(Rank::ACE, Suit::SPADES));
  hand.addCard(Card(Rank::SIX, Suit::HEARTS));

  EXPECT_EQ(hand.getTotal(), 17);
  EXPECT_TRUE(hand.isSoft());
}

TEST_F(HandTest, HardAceHandling) {
  // Ace + 6 + 9 = 16 (hard, ace becomes 1)
  hand.addCard(Card(Rank::ACE, Suit::SPADES));
  hand.addCard(Card(Rank::SIX, Suit::HEARTS));
  hand.addCard(Card(Rank::NINE, Suit::CLUBS));

  EXPECT_EQ(hand.getTotal(), 16);
  EXPECT_FALSE(hand.isSoft());
}

TEST_F(HandTest, MultipleAcesHandling) {
  // Ace + Ace + 9 = 21 (one ace is 11, other is 1)
  hand.addCard(Card(Rank::ACE, Suit::SPADES));
  hand.addCard(Card(Rank::ACE, Suit::HEARTS));
  hand.addCard(Card(Rank::NINE, Suit::CLUBS));

  EXPECT_EQ(hand.getTotal(), 21);
  EXPECT_TRUE(hand.isSoft());
}

TEST_F(HandTest, BlackjackDetection) {
  // Ace + 10 = Blackjack
  hand.addCard(Card(Rank::ACE, Suit::SPADES));
  hand.addCard(Card(Rank::TEN, Suit::HEARTS));

  EXPECT_TRUE(hand.isBlackjack());
  EXPECT_EQ(hand.getTotal(), 21);
}

TEST_F(HandTest, TwentyOneNotBlackjack) {
  // Three cards totaling 21 is not blackjack
  hand.addCard(Card(Rank::SEVEN, Suit::SPADES));
  hand.addCard(Card(Rank::SEVEN, Suit::HEARTS));
  hand.addCard(Card(Rank::SEVEN, Suit::CLUBS));

  EXPECT_FALSE(hand.isBlackjack());
  EXPECT_EQ(hand.getTotal(), 21);
}

TEST_F(HandTest, BustDetection) {
  hand.addCard(Card(Rank::KING, Suit::SPADES));
  hand.addCard(Card(Rank::QUEEN, Suit::HEARTS));
  hand.addCard(Card(Rank::FIVE, Suit::CLUBS));

  EXPECT_TRUE(hand.isBust());
  EXPECT_EQ(hand.getTotal(), 25);
}

TEST_F(HandTest, CanSplitPairs) {
  hand.addCard(Card(Rank::EIGHT, Suit::SPADES));
  hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));

  EXPECT_TRUE(hand.canSplit());
}

TEST_F(HandTest, CannotSplitNonPairs) {
  hand.addCard(Card(Rank::EIGHT, Suit::SPADES));
  hand.addCard(Card(Rank::NINE, Suit::HEARTS));

  EXPECT_FALSE(hand.canSplit());
}

TEST_F(HandTest, CanSplitFaceCards) {
  // Different face cards can still split (both worth 10)
  hand.addCard(Card(Rank::JACK, Suit::SPADES));
  hand.addCard(Card(Rank::JACK, Suit::HEARTS));

  EXPECT_TRUE(hand.canSplit());
}

TEST_F(HandTest, ClearHand) {
  hand.addCard(Card(Rank::FIVE, Suit::HEARTS));
  hand.addCard(Card(Rank::SEVEN, Suit::SPADES));

  EXPECT_EQ(hand.size(), 2);

  hand.clear();

  EXPECT_TRUE(hand.empty());
  EXPECT_EQ(hand.getTotal(), 0);
}

TEST_F(HandTest, SplitHandReturnsSecondCard) {
  Card first(Rank::EIGHT, Suit::SPADES);
  Card second(Rank::EIGHT, Suit::HEARTS);

  hand.addCard(first);
  hand.addCard(second);

  Card returned = hand.split();

  EXPECT_EQ(returned, second);
  EXPECT_EQ(hand.size(), 1);
  EXPECT_EQ(hand.getCards()[0], first);
}