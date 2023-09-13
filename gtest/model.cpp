#include <automodel.hpp>
#include <funcs.hpp>
#include <vector>

#include "dataset.hpp"
#include "include/st2013.hpp"
#include "gtest/gtest.h"

using namespace std;

model::base_lattice chain(size_t N) {
  std::vector<size_t> shapes = {N};
  return model::base_lattice("chain lattice", "simple1d", shapes,
                             "../config/lattices.xml", false);
}

model::base_lattice chain10 = chain(10);
std::vector<std::vector<std::vector<double>>> heisenberg1D_hams = {
    heisenberg1D_ham};

// TEST(ModelTest, ReadNpyHeisenberg1D){
//   vector<size_t> dofs = {2};
//   std::vector<double> params = {1.0};
//   std::vector<int> types = {0};
//   double shift = 0.25;
//
//   model::base_model<bcl::st2013> spin(chain10, dofs, heisenberg1D_hams,
//   shift, false);
//
//   model::base_model<bcl::st2013> spin2(chain10, dofs,
//         "../gtest/model_array/Heisenberg/original/Jz_1.0_Jx_1.0_h_0.0/H",
//         params, types, shift, false, false, false);
//
//   // cout << spin2.loperators[0].ham_vector() << endl;
//
//   // cout << "-------------------------------" << endl;
//
//   // cout << spin2.loperators[0].ham() << endl;
//
//   EXPECT_EQ(spin.loperators[0], spin2.loperators[0]);
// }
//
// TEST(ModelTest, Heisenberg1D) {
//   vector<size_t> dofs = {2};
//   std::vector<double> params = {1.0};
//   std::vector<int> types = {0};
//   double shift = 0.25;
//   model::base_model<bcl::st2013> spin(chain10, dofs, heisenberg1D_hams,
//   shift, false);
//   // cerr << heisenberg1D_hams << endl;
//
//   EXPECT_EQ(1, spin.N_op);
//   EXPECT_EQ(spin.N_op, spin.loperators.size());
//
//   ASSERT_EQ(heisenberg1D_ham_vector.size(),
//   spin.loperators[0].ham_vector().size());
//
//   for (size_t i=0; i<heisenberg1D_ham_vector.size(); i++){
//     EXPECT_EQ(heisenberg1D_ham_vector[i],
//     spin.loperators[0].ham_vector()[i]);
//   }
// }

TEST(ModelTest, HeisenbergUnitary) {
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  model::base_model<bcl::st2013> spin(
      chain10, dofs,
      "../gtest/model_array/Heisenberg/1D/original/"
      "Jx_-0.3_Jy_0.5_Jz_0.8_hx_0.3_hz_0/H",
      "../gtest/model_array/Heisenberg/1D/original/mes/"
      "Jx_-0.3_Jy_0.5_Jz_0.8_hx_0.3_hz_0/M_10/u",
      params, types, shift, false, false, false);
  model::base_model<bcl::st2013> spin2(
      chain10, dofs,
      "../gtest/model_array/Heisenberg/1D/original/mes/"
      "Jx_-0.3_Jy_0.5_Jz_0.8_hx_0.3_hz_0/M_10/H",
      params, types, shift, false, false, false);

  // check if hamiltonian is the same
  auto h1 = spin.loperators[0].ham();
  auto h2 = spin2.loperators[0].ham();

  for (size_t i = 0; i < h1.size(); i++) {
    for (size_t j = 0; j < h1[i].size(); j++) {
      EXPECT_NEAR(h1[i][j], h2[i][j], 1E-8);
    }
  }
}

std::vector<size_t> shapes = {2, 2};
model::base_lattice lat("triangular lattice", "anisotropic triangular", shapes,
                        "../config/lattices.xml", false);

TEST(ModelTest, KagomeUnitary) {
  string ham_path_mes =
      "../gtest/model_array/KH/mes/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_100/H/";
  string u_path =
      "../gtest/model_array/KH/mes/Jx_1_Jy_1_Jz_1_hx_0_hz_0/M_100/u/";
  string ham_path_none =
      "../gtest/model_array/KH/none/Jx_1_Jy_1_Jz_1_hx_0_hz_0/H/";

  model::base_model<bcl::st2013> spin(lat, {8}, ham_path_none, u_path,
                                      {1, 1, 1}, {0, 1, 2}, 0.1, false, false,
                                      false, 0);
  model::base_model<bcl::st2013> spin2(lat, {8}, ham_path_mes, {1, 1, 1},
                                       {0, 1, 2}, 0.1, false, false, false, 0);


  for (size_t n = 0; n < spin.N_op; n++)
  {
    auto h1 = spin.loperators[n].ham();
    auto h2 = spin2.loperators[n].ham();
    for (size_t i = 0; i < h1.size(); i++) {
      for (size_t j = 0; j < h1[i].size(); j++) {
        EXPECT_NEAR(h1[i][j], h2[i][j], 1E-8);
      }
    }
  }
}
