#include <automodel.hpp>
#include <vector>
#include <string>

#include <funcs.hpp>
#include <state.hpp>
#include "gtest/gtest.h"
#include "dataset.hpp"

using namespace std;

TEST(HamsTest, Kagome_2x2_test) {
  std::vector<size_t> shapes = {2, 2};
  model::base_lattice lat("triangular lattice", "anisotropic triangular", shapes, "../config/lattices.xml", false);
  string ham_path = "../gtest/model_array/KH/smel/H1";
  model::base_model<bcl::st2013> spin(lat, {8}, ham_path, {1,1,1}, {0,1,2}, 0.1, false, false, true);
  vector<vector<double>> H = spin.loperators[0]._ham;
  // cerr << H[0] << endl;
  //* diagonal element at site = (0, 1) spins are = (1, 0)

  spin_state::StateFunc state_func(8, 2);
  int x = state_func.state2num({1,0},2);

  //! Note that the indexing rule of numpy is different from this.
  //! In numpy, {1,0} will be 8 but here it is 1.
  EXPECT_EQ(x, 1);
  EXPECT_FLOAT_EQ(H[x][x], 0.16666234482156317);

  //d* test single flip operator
  //* end spin is fixed to 3
  auto single_flip = spin.loperators[0].single_flip(0, 3);

  double shift = 0;
  for (int i=0; i<H.size();i++){
    shift = std::min(shift, H[i][i]);
  }
  shift *= -1;
  shift += 0.1;
  EXPECT_FLOAT_EQ(shift, spin.loperators[0].ene_shift);
  EXPECT_FLOAT_EQ(single_flip[1][3], -0.11785105507860531);

  //d* diagonal element is little bit tricky
  double elem2 = -0.16666596146430332;
  elem2 += shift;
  elem2 = abs(elem2);
  double alpha = 1 / 6.0;
  EXPECT_FLOAT_EQ(spin.loperators[0].ham_prime[2 + 3 * 8][2 + 3 * 8], elem2);
  EXPECT_FLOAT_EQ(single_flip[2][2], elem2 / 2 * alpha);

  //* fix end site
  auto single_flip2 = spin.loperators[0].single_flip(1, 2);
  EXPECT_FLOAT_EQ(single_flip2[1][3],  -0.11785629850799827);
  elem2 =  -0.33333079565882834;
  elem2 += shift;
  elem2 = abs(elem2);
  EXPECT_FLOAT_EQ(spin.loperators[0].ham_prime[2 + 4 * 8][2 + 4 * 8], elem2);
  EXPECT_FLOAT_EQ(single_flip2[4][4], elem2 / 2 * alpha);
}



