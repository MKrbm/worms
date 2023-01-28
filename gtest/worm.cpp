#include <worm.hpp>

#include <observable.hpp>
#include <iostream>
#include <random>
#include "gtest/gtest.h"

TEST(WormTest, Ising) {
  BC::observable obs;
  obs << 1.0;
  obs << 2.0;
  obs << 3.0;
  EXPECT_FLOAT_EQ(2.0, obs.mean());
}



