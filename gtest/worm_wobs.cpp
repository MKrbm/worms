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

typedef bcl::st2013 MC;
Worm<MC> run_worm(
  model::base_model<MC>& spin, 
  double T, size_t sweeps, size_t therms,
  std::vector<batch_res> &res, 
  model::observable &obs,
  model::base_lattice &lat,
  model::WormObs wobs = model::WormObs(2)
  ){
    //dont fix worm density. Not printout density information.
  Worm<MC> solver = exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, obs, wobs);

  batch_res as = res[0]; // average sign 
  batch_res ene = res[1]; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res n_neg_ele = res[2];
  batch_res n_ops = res[3];
  batch_res N2 = res[4];
  batch_res N = res[5];
  batch_res dH = res[6]; // $\frac{\frac{\partial}{\partial h}Z}{Z}$ 
  batch_res dH2 = res[7]; // $\frac{\frac{\partial^2}{\partial h^2}Z}{Z}$
  batch_res worm_obs = res[8];
  batch_res phys_conf = res[9];
  batch_res m2_diag = res[10];

  std::function<double(double, double, double)> f;

  pair<double, double> as_mean = jackknife_reweight_single(as);          // calculate <S>
  pair<double, double> nop_mean = jackknife_reweight_single(n_ops);      // calculate <S>
  pair<double, double> nnop_mean = jackknife_reweight_single(n_neg_ele); // calculate <S>

  // calculate energy
  pair<double, double> ene_mean = jackknife_reweight_div(ene, as); // calculate <SH> / <S>

  // calculate worm_observable 
  if (phys_conf.mean()[0] == 0) {throw std::runtime_error("No physical configuration");}
  pair<double, double> worm_obs_mean = jackknife_reweight_div(worm_obs, phys_conf);  // calculate <WoS> / <S>
  
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
       << endl;
  cout << "-------------------------------------------------" << endl;
  cout << "Energy per site      = "
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
       << chi_mean.first  * T / lat.L << " +- " << chi_mean.second  * T / lat.L << endl
       << "G                    = "
       << worm_obs_mean.first << " +- " << worm_obs_mean.second << endl
       << "worm obs mean        = "
       << worm_obs.mean()[0] << " +- " << worm_obs.mean()[1] << endl
       << "Physical configurations = "
       << phys_conf.mean()[0] << " +- " << phys_conf.mean()[1] << endl;
  return solver;
}


TEST(WormSimuObs, HXXX1D_1) {

  model::base_lattice lat = chain(10);
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 5;
  bool zw = true;

  std::string ham_path, obs_path, wobs_path;

  // This hamiltonian requires zero worm.
  ham_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_0.5_Jy_0.3_hz_0_hx_0.5/H";
  obs_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_0.5_Jy_0.3_hz_0_hx_0.5/Sz";
  wobs_path = "../gtest/model_array/worm_obs/g_test";
  model::base_model<MC> spin(lat, dofs, 
  ham_path, params, types, shift, zw, false, false);
  // cerr << heisenberg1D_hams << endl;

  model::observable obs(spin,obs_path , false);
  model::WormObs wobs(spin.sps_sites(0), wobs_path);
  vector<batch_res> res;
  double T;
  size_t sweeps, therms;

  sweeps = 5000000;
  T = 0.5;
  therms = 100000;

  auto solver = run_worm(spin, T, sweeps, therms, res, obs, lat, wobs);

  /*
  expect the following res 

  L = 10 {'Jz': -1, 'Jx': 0.5, 'Jy': 0.3, 'hz': 0, 'hx': 0.5}
  T               = 0.5
  E               = -0.17920077733642997
  C               = 0.1904521933525169
  M               = 2.2011215518437648e-17
  chi             = 1.1863488077976516
  G_test          = 0.7632810332661033
  */
}



TEST(WormSimuObs, HXXX1D_0) {

  model::base_lattice lat = chain(12);
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.25;
  bool zw = false;

  std::string ham_path, obs_path, wobs_path;
  ham_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_-1_Jy_-1_h_0/H";
  obs_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_-1_Jy_-1_h_0/Sz";
  wobs_path = "../gtest/model_array/worm_obs/g_nsingle";
  model::base_model<MC> spin(lat, dofs, 
  ham_path, params, types, shift, zw, false, false);
  // cerr << heisenberg1D_hams << endl;

  model::observable obs(spin,obs_path , false);
  model::WormObs wobs(spin.sps_sites(0), wobs_path);
  vector<batch_res> res;
  double T;
  size_t sweeps, therms;

  sweeps = 1000000;
  T = 1;
  therms = 100000;

  auto solver = run_worm(spin, T, sweeps, therms, res, obs, lat, wobs);

  /*
  expect the following res 

  L = 12 {'Jz': -1, 'Jx': -1, 'Jy': -1, 'h': 0}
  T               = 1
  E               = -0.13407600893675187
  C               = 0.08387609895214143
  M               = 2.0710757781922097e-17
  M^2             = 0.03068815417665093
  Chi             = 0.36824933153474
  G4(No one site) = 0.6309046165820786
  */
}


