#include <automodel.hpp>
#include <vector>
#include <string>

#include <funcs.hpp>
#include "gtest/gtest.h"

using namespace std;

TEST(LatticeTest, Simple1D) {
  std::vector<size_t> shapes = {10};
  model::base_lattice lat("chain lattice", "simple1d", shapes, "../config/lattices.xml", false);
  EXPECT_EQ(10, lat.L);
}


TEST(LatticeTest, Simple2D) {
  std::vector<size_t> shapes = {10, 5};
  model::base_lattice lat("square lattice", "simple2d", shapes, "../config/lattices.xml", false);
  EXPECT_EQ(10*5, lat.L);
  EXPECT_EQ(lat.L * 2, lat.bonds.size());

  std::vector<size_t> shapes2 = {10, 5, 3};
  EXPECT_DEATH(model::base_lattice("square lattice", "simple2d", shapes2, "../config/lattices.xml", false),
    "Wrong number of shapes for 2D lattice");
}

TEST(LatticeTest, Kagome) {
  std::vector<size_t> shapes = {10, 5};
}


