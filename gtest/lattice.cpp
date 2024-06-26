#include <automodel.hpp>
#include <vector>
#include <string>

#include <funcs.hpp>
#include "gtest/gtest.h"
#include "dataset.hpp"

using namespace std;

TEST(LatticeTest, Kagome_2x2) {
  std::vector<size_t> shapes = {2, 2};
  model::base_lattice lat("triangular lattice", "kagome", shapes, "../config/lattices.xml", false);
  cout << lat.bonds << endl;
  for (size_t i = 0; i < lat.L; i++) {
    EXPECT_EQ(2, lat.bonds[i].size());
    for (size_t j = 0; j < lat.bonds[i].size(); j++) {
      EXPECT_EQ(kagome_bonds[i][j], lat.bonds[i][j]);
    }
  }
}

TEST(LatticeTest, Kagome_3x2) {
  std::vector<size_t> shapes = {3, 1};
  model::base_lattice lat("triangular lattice", "kagome", shapes, "../config/lattices.xml", true);
  cout << lat.bonds << endl;
}

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




