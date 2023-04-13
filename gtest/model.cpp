#include <automodel.hpp>
#include <vector>
#include <funcs.hpp>

#include "dataset.hpp"
#include "gtest/gtest.h"

using namespace std;

model::base_lattice chain(size_t N){
  std::vector<size_t> shapes = {N};
  return model::base_lattice("chain lattice", "simple1d", shapes, "../config/lattices.xml", false);
}

model::base_lattice chain10 = chain(10);
std::vector<std::vector<std::vector<double>>> heisenberg1D_hams = {heisenberg1D_ham};

TEST(ModelTest, ReadNpyHeisenberg1D){
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25; 

  model::base_model<bcl::st2013> spin(chain10, dofs, heisenberg1D_hams, shift, false);

  model::base_model<bcl::st2013> spin2(chain10, dofs, 
        "../gtest/model_array/Heisenberg/original/Jz_1.0_Jx_1.0_h_0.0/H", 
        params, types, shift, false, false, false);

  // cout << spin2.loperators[0].ham_vector() << endl;

  // cout << "-------------------------------" << endl;

  // cout << spin2.loperators[0].ham() << endl;

  EXPECT_EQ(spin.loperators[0], spin2.loperators[0]);
}

TEST(ModelTest, Heisenberg1D) {
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  model::base_model<bcl::st2013> spin(chain10, dofs, heisenberg1D_hams, shift, false);
  // cerr << heisenberg1D_hams << endl;
  
  EXPECT_EQ(1, spin.N_op);
  EXPECT_EQ(spin.N_op, spin.loperators.size());

  ASSERT_EQ(heisenberg1D_ham_vector.size(), spin.loperators[0].ham_vector().size());

  for (size_t i=0; i<heisenberg1D_ham_vector.size(); i++){
    EXPECT_EQ(heisenberg1D_ham_vector[i], spin.loperators[0].ham_vector()[i]);
  }
}