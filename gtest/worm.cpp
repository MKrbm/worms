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

model::base_lattice chain10 = chain(10);
std::vector<std::vector<std::vector<double>>> heisenberg1D_hams = {heisenberg1D_ham};



TEST(WormTest, Heisenberg1D) {


  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  model::base_model<bcl::st2013> spin(chain10, dofs, heisenberg1D_hams, shift, false);
  // cerr << heisenberg1D_hams << endl;

  EXPECT_EQ(1, spin.N_op);
  EXPECT_EQ(spin.N_op, spin.loperators.size());

  ASSERT_EQ(heisenberg1D_ham_vector.size(), spin.loperators[0].ham_vector().size());

  for (size_t i = 0; i < heisenberg1D_ham_vector.size(); i++)
  {
    EXPECT_EQ(heisenberg1D_ham_vector[i], spin.loperators[0].ham_vector()[i]);
  }

  model::observable obs(spin, "", false);
  vector<batch_res> res;

  double T;
  size_t sweeps, therms;

  T = 1;
  sweeps = 1000000;
  therms = 100000;


  //dont fix worm density. Not printout density information.
  exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, obs);

  batch_res as = res[0];  // average sign
  batch_res ene = res[1]; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res sglt = res[2];
  batch_res n_neg_ele = res[3];
  batch_res n_ops = res[4];
  batch_res N2 = res[5];
  batch_res N = res[6];
  batch_res dH = res[7];  // $\frac{\frac{\partial}{\partial h}Z}{Z}$
  batch_res dH2 = res[8]; // $\frac{\frac{\partial^2}{\partial h^2}Z}{Z}$

  std::function<double(double, double, double)> f;

  pair<double, double> as_mean = jackknife_reweight_single(as);          // calculate <S>
  pair<double, double> nop_mean = jackknife_reweight_single(n_ops);      // calculate <S>
  pair<double, double> nnop_mean = jackknife_reweight_single(n_neg_ele); // calculate <S>

  // calculate energy
  pair<double, double> ene_mean = jackknife_reweight_div(ene, as); // calculate <SH> / <S>

  // calculat heat capacity
  f = [](double x1, double x2, double y)
  { return (x2 - x1) / y - (x1 / y) * (x1 / y); };
  pair<double, double> c_mean = jackknife_reweight_any(N, N2, as, f);

  // calculate magnetization
  pair<double, double> m_mean = jackknife_reweight_div(dH, as);

  // calculate susceptibility
  f = [](double x1, double x2, double y)
  { return x2 / y - (x1 / y) * (x1 / y); };
  pair<double, double> chi_mean = jackknife_reweight_any(dH, dH2, as, f);

  cout << chain10.L << endl;
  cout << "temperature          = " << T
       << endl
       << "Average sign         = "
       << as_mean.first << " +- " 
       << as_mean.second   
       << endl
       << "Energy per site      = "
       << ene_mean.first / chain10.L << " +- "
       << ene_mean.second / chain10.L
       << endl;
}


TEST(WormTest, Heisenberg2D) {

  std::vector<size_t> shapes = {4, 4};
  model::base_lattice square4("square lattice", "simple2d", shapes, "../config/lattices.xml", false);


  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  bool zw = false;
  model::base_model<bcl::st2013> spin(square4, dofs, "../gtest/model_array/Heisenberg/original/Jz_1.0_Jx_1.0_h_0.0/H", 
        params, types, shift, zw, false, false); //heisenberg with J = 1, h =0 (H = J \sum S\cdot S - h)



  model::observable obs(spin, "", false);
  vector<batch_res> res;

  double T;
  size_t sweeps, therms;

  T = 1;
  sweeps = 1000000;
  therms = 100000;


  //dont fix worm density. Not printout density information.
  exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, obs);

  batch_res as = res[0];  
  batch_res ene = res[1]; 
  batch_res sglt = res[2];
  batch_res n_neg_ele = res[3];
  batch_res n_ops = res[4];
  batch_res N2 = res[5];
  batch_res N = res[6];
  batch_res dH = res[7];  
  batch_res dH2 = res[8]; 

  std::function<double(double, double, double)> f;

  pair<double, double> as_mean = jackknife_reweight_single(as);          // calculate <S>
  pair<double, double> nop_mean = jackknife_reweight_single(n_ops);      // calculate <S>
  pair<double, double> nnop_mean = jackknife_reweight_single(n_neg_ele); // calculate <S>

  // calculate energy
  pair<double, double> ene_mean = jackknife_reweight_div(ene, as); // calculate <SH> / <S>

  // calculat heat capacity
  f = [](double x1, double x2, double y)
  { return (x2 - x1) / y - (x1 / y) * (x1 / y); };
  pair<double, double> c_mean = jackknife_reweight_any(N, N2, as, f);

  // calculate magnetization
  pair<double, double> m_mean = jackknife_reweight_div(dH, as);

  // calculate susceptibility
  f = [](double x1, double x2, double y)
  { return x2 / y - (x1 / y) * (x1 / y); };
  pair<double, double> chi_mean = jackknife_reweight_any(dH, dH2, as, f);

  cout << square4.L << endl;
  cout << "temperature          = " << T
       << endl
       << "Average sign         = "
       << as_mean.first << " +- " 
       << as_mean.second   
       << endl
       << "Energy per site      = "
       << ene_mean.first / square4.L << " +- "
       << ene_mean.second / square4.L
       << endl;
}


TEST(WormTest, Heisenberg2DNS) { // with negative sign

  std::vector<size_t> shapes = {4, 4};
  model::base_lattice square4("square lattice", "simple2d", shapes, "../config/lattices.xml", false);


  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  bool zw = false;
  model::base_model<bcl::st2013> spin(square4, dofs, "../gtest/model_array/Heisenberg/original/Jz_1.0_Jx_1.0_h_0.0/H", 
        params, types, shift, zw, false, false); //heisenberg with J = 1, h =0 (H = J \sum S\cdot S - h)



  model::observable obs(spin, "", false);
  vector<batch_res> res;

  double T;
  size_t sweeps, therms;

  T = 1;
  sweeps = 1000000;
  therms = 100000;


  //dont fix worm density. Not printout density information.
  exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, obs);

  batch_res as = res[0];  
  batch_res ene = res[1]; 
  batch_res sglt = res[2];
  batch_res n_neg_ele = res[3];
  batch_res n_ops = res[4];
  batch_res N2 = res[5];
  batch_res N = res[6];
  batch_res dH = res[7];  
  batch_res dH2 = res[8]; 

  std::function<double(double, double, double)> f;

  pair<double, double> as_mean = jackknife_reweight_single(as);          // calculate <S>
  pair<double, double> nop_mean = jackknife_reweight_single(n_ops);      // calculate <S>
  pair<double, double> nnop_mean = jackknife_reweight_single(n_neg_ele); // calculate <S>

  // calculate energy
  pair<double, double> ene_mean = jackknife_reweight_div(ene, as); // calculate <SH> / <S>

  // calculat heat capacity
  f = [](double x1, double x2, double y)
  { return (x2 - x1) / y - (x1 / y) * (x1 / y); };
  pair<double, double> c_mean = jackknife_reweight_any(N, N2, as, f);

  // calculate magnetization
  pair<double, double> m_mean = jackknife_reweight_div(dH, as);

  // calculate susceptibility
  f = [](double x1, double x2, double y)
  { return x2 / y - (x1 / y) * (x1 / y); };
  pair<double, double> chi_mean = jackknife_reweight_any(dH, dH2, as, f);

  cout << square4.L << endl;
  cout << "temperature          = " << T
       << endl
       << "Average sign         = "
       << as_mean.first << " +- " 
       << as_mean.second   
       << endl
       << "Energy per site      = "
       << ene_mean.first / square4.L << " +- "
       << ene_mean.second / square4.L
       << endl;
}





