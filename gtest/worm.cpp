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

std::vector<std::vector<std::vector<double>>> heisenberg1D_hams = {heisenberg1D_ham};


void run_worm(
  model::base_model<bcl::st2013>& spin, 
  double T, size_t sweeps, size_t therms,
  std::vector<batch_res> &res, 
  model::observable &obs,
  model::base_lattice &lat
  ){
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

  cout << lat.L << endl;
  cout << "temperature          = " << T
       << endl
       << "Average sign         = "
       << as_mean.first << " +- " 
       << as_mean.second   
       << endl
       << "Energy per site      = "
       << ene_mean.first / lat.L << " +- "
       << ene_mean.second / lat.L
       << endl
       << "Specific heat        = "
       << c_mean.first / lat.L << " +- " 
       << c_mean.second / lat.L
       << endl
       << "magnetization        = "
       << m_mean.first * T / lat.L << " +- " << m_mean.second * T / lat.L
       << endl
       << "susceptibility       = "
       << chi_mean.first  * T / lat.L << " +- " << chi_mean.second  * T / lat.L << endl;
}


TEST(WormTest, Heisenberg1D) {

  model::base_lattice lat = chain(9);
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  bool zw = false;
  model::base_model<bcl::st2013> spin(lat, dofs, "../gtest/model_array/Heisenberg/1D/original/Jz_1_Jx_1_h_1/H", params, types, shift, zw, false, false);
  // cerr << heisenberg1D_hams << endl;

  model::observable obs(spin, "../gtest/model_array/Heisenberg/1D/original/Jz_1_Jx_1_h_1/Sz", false);
  vector<batch_res> res;

  double T;
  size_t sweeps, therms;

  T = 1;
  sweeps = 1000000;
  therms = 100000;

  run_worm(spin, T, sweeps, therms, res, obs, lat);

  /*
  expect the following res

  T   = 1
  E   = -0.29693777450469444
  C   = 0.1778936802509913
  M   = 0.1366365772892668
  chi = 0.13636810663606222
  */
}

TEST(WormTest, Heisenberg1D2) {

  model::base_lattice lat = chain(9);
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  bool zw = false;
  model::base_model<bcl::st2013> spin(lat, dofs, "../gtest/model_array/Heisenberg/1D/original/Jz_1_Jx_-0.3_Jy_1_h_0.25/H", params, types, shift, zw, false, false);
  // cerr << heisenberg1D_hams << endl;

  model::observable obs(spin, "../gtest/model_array/Heisenberg/1D/original/Jz_1_Jx_-0.3_Jy_1_h_0.25/Sz", false);
  vector<batch_res> res;

  double T;
  size_t sweeps, therms;

  T = 0.4;
  sweeps = 1000000;
  therms = 100000;

  run_worm(spin, T, sweeps, therms, res, obs, lat);

  /*
  expect the following res
  T   = 1
  E   = -0.11794811240370676
  C   = 0.08996577957529565
  M   = 0.03730406777171547
  chi = 0.1613498985228055
  */
}



TEST(WormTest, Heisenberg2D) {

  std::vector<size_t> shapes = {4, 4};
  model::base_lattice lat("square lattice", "simple2d", shapes, "../config/lattices.xml", false);


  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  bool zw = false;
  model::base_model<bcl::st2013> spin(lat, dofs, "../gtest/model_array/Heisenberg/original/Jz_1.0_Jx_1.0_h_0.0/H", 
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

  cout << lat.L << endl;
  cout << "temperature          = " << T
       << endl
       << "Average sign         = "
       << as_mean.first << " +- " 
       << as_mean.second   
       << endl
       << "Energy per site      = "
       << ene_mean.first / lat.L << " +- "
       << ene_mean.second / lat.L
       << endl;
}


TEST(WormTest, Heisenberg2DNS) { // with negative sign

  std::vector<size_t> shapes = {3, 3};
  model::base_lattice lat("square lattice", "simple2d", shapes, "../config/lattices.xml", false);


  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  bool zw = false;
  model::base_model<bcl::st2013> spin(lat, dofs, "../gtest/model_array/Heisenberg/2D/original/Jz_1.0_Jx_1.0_h_1.0/H", 
        params, types, shift, zw, false, false); //heisenberg with J = 1, h =0 (H = J \sum S\cdot S - h)



  model::observable obs(spin, "../gtest/model_array/Heisenberg/2D/original/Jz_1.0_Jx_1.0_h_1.0/Sz", false);
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


  cout << lat.L << endl;
  cout << "temperature          = " << T
       << endl
       << "Average sign         = "
       << as_mean.first << " +- " 
       << as_mean.second   
       << endl
       << "Energy per site      = "
       << ene_mean.first / lat.L << " +- "
       << ene_mean.second / lat.L
       << endl
       << "Specific heat        = "
       << c_mean.first / lat.L << " +- " 
       << c_mean.second / lat.L
       << endl
       << "magnetization        = "
       << m_mean.first * T / lat.L << " +- " << m_mean.second * T / lat.L
       << endl
       << "susceptibility       = "
       << chi_mean.first  * T / lat.L << " +- " << chi_mean.second  * T / lat.L << endl;

   /*
   should be
   temperature            = 1.000000000000001
   energy(per site)      =  -0.36435048892860666 += 0.00273718093728829
   specific heat         =  0.17724811597506396 += 0.018925580787546897
   magnetization(FTLM)   =  0.10177510995282293 += 0.0015859477475889897
   magnetization(LTLM)   =  0.10177510995282289 += 0.0016238244565489892
   suceptibility(FTLM)   =  0.19129067493962343 += -0.00047779199410538534
   suceptibility(LTLM)   =  0.19129067493962346 += -0.00047780518430885023
   */
}





