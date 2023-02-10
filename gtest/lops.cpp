#include <mpi.h>
#include <vector>
#include <funcs.hpp>

#include <alps/alea/batch.hpp>
#include <alps/utilities/mpi.hpp>
#include <alps/alea/mpi.hpp>

#include "exec_parallel.hpp"
#include <automodel.hpp>
#include <jackknife.hpp>

#include "dataset.hpp"
#include "gtest/gtest.h"

double elapsed;

using namespace std;


model::base_lattice chain(size_t N)
{
  std::vector<size_t> shapes = {N};
  return model::base_lattice("chain lattice", "simple1d", shapes, "../config/lattices.xml", false);
}

typedef bcl::st2013 MC;

TEST(Lops, warp) {
  model::base_lattice lat = chain(12);
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  bool zw = true;

  std::string ham_path, obs_path, wobs_path;
  ham_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_-0.5_Jy_-0.3_hz_0_hx_0.5/H";
  obs_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_-0.5_Jy_-0.3_hz_0_hx_0.5/Sz";
  wobs_path = "../gtest/model_array/worm_obs/gtest";
  model::base_model<MC> spin(lat, dofs, ham_path, params, types, shift, zw, false, false);
  EXPECT_EQ(spin.loperators.size(), 1);
  local_operator<MC> lop = spin.loperators[0];
  for (int i=0; i<lop.ham_vector().size(); i++){
    EXPECT_FLOAT_EQ(check_warp_ham[i], lop.ham_vector(i));
  }
  EXPECT_EQ(lop.has_warp(0), true);

  zw = false;
  model::base_model<MC> spin_prime(lat, dofs, ham_path, params, types, shift, zw, false, false);
  EXPECT_EQ(spin_prime.loperators[0].has_warp(0), false);



  ham_path = "../gtest/model_array/Heisenberg/1D/original/Jz_1_Jx_-0.3_Jy_0.5_h_1/H";
  obs_path = "../gtest/model_array/Heisenberg/1D/original/Jz_1_Jx_-0.3_Jy_0.5_h_1/Sz";

  model::base_model<MC> spin2(lat, dofs, ham_path, params, types, shift, zw, false, false);
  local_operator<MC> lop2 = spin2.loperators[0];
  for (int i=0; i<lop2.ham_vector().size(); i++){
    EXPECT_EQ(lop2.has_warp(i), false);
  }

  ham_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_0_Jy_0_hz_0_hx_1/H";
  obs_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_0_Jy_0_hz_0_hx_1/Sz";
  shift = 0;
  zw = true;
  model::base_model<MC> spin3(lat, dofs, ham_path, params, types, shift, zw, false, false);
  local_operator<MC> lop3 = spin3.loperators[0];
  for (int i=0; i<lop3.ham_vector().size(); i++){
    EXPECT_EQ(lop3.has_warp(i), check_has_warp[i]);
  }
}


