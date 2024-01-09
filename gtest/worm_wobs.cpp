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
void run_worm(
    model::base_model<MC> &spin,
    double T, size_t sweeps, size_t therms,
    std::vector<batch_res> &res,
    model::observable &obs,
    model::base_lattice &lat,
    model::MapWormObs wobs = model::WormObs(2),
    size_t n_sites = 1)
{
  size_t ns = lat.L * n_sites;
  alps::alea::autocorr_result<double> ac_res;
  // dont fix worm density. Not printout density information.
  double r;
  auto solver = exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, ac_res, obs, wobs, r);

  batch_res as = res[0];  // average sign
  batch_res ene = res[1]; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res n_neg_ele = res[2];
  batch_res n_ops = res[3];
  batch_res N2 = res[4];
  batch_res N = res[5];
  batch_res dH = res[6];  // $\frac{\frac{\partial}{\partial h}Z}{Z}$
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
  if (phys_conf.mean()[0] == 0)
  {
    throw std::runtime_error("No physical configuration");
  }
  pair<double, double> worm_obs_mean = jackknife_reweight_div(worm_obs, phys_conf); // calculate <WoS> / <S>

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

  cout << ns << endl;
  cout << "temperature          = " << T
       << endl
       << "Average sign         = "
       << as_mean.first << " +- "
       << as_mean.second
       << endl;
  cout << "-------------------------------------------------" << endl;
  cout << "Energy per site      = "
       << ene_mean.first / ns << " +- "
       << ene_mean.second / ns
       << endl
       << "Specific heat        = "
       << c_mean.first / ns << " +- "
       << c_mean.second / ns
       << endl
       << "magnetization        = "
       << m_mean.first * T / ns << " +- " << m_mean.second * T / ns
       << endl
       << "susceptibility       = "
       << chi_mean.first * T / ns << " +- " << chi_mean.second * T / ns << endl
       << "G                    = "
       << worm_obs_mean.first << " +- " << worm_obs_mean.second << endl
       << "worm obs mean        = "
       << worm_obs.mean()[0] << " +- " << worm_obs.mean()[1] << endl
       << "Physical configurations = "
       << phys_conf.mean()[0] << " +- " << phys_conf.mean()[1] << endl;
  return;
}

TEST(WormSimuObs, HXXX1D_NS)
{

  model::base_lattice lat = chain(10);
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.1;
  bool zw = true;

  std::string ham_path, obs_path, wobs_path;

  // This hamiltonian requires zero worm.
  ham_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_0.5_Jy_0.3_hz_0_hx_0.5/H";
  obs_path = "../gtest/model_array/Heisenberg/1D/original/Jz_-1_Jx_0.5_Jy_0.3_hz_0_hx_0.5/Sz";
  wobs_path = "../gtest/model_array/worm_obs/g_test";
  model::base_model<MC> spin(lat, dofs,
                             ham_path, params, types, shift, zw, false, false);
  // cerr << heisenberg1D_hams << endl;

  model::observable obs(spin, obs_path, false);
  model::WormObs wobs(spin.sps_sites(0), wobs_path);
  vector<batch_res> res;
  double T;
  size_t sweeps, therms;

  sweeps = 5000000;
  T = 0.5;
  therms = 100000;

  run_worm(spin, T, sweeps, therms, res, obs, lat, wobs);

  /*
  expect the following res

  L = 10 {'Jz': -1, 'Jx': 0.5, 'Jy': 0.3, 'hz': 0, 'hx': 0.5}
  T               = 0.5
  E               = -0.17920077733642997
  C               = 0.1904521933525169
  M               = 2.2011215518437648e-17
  chi             = 1.1863488077976516
  G_test          = 5.013504505157471


  N = 10,000,000
  Elapsed time         = 62.037(62.037) sec
  Speed                = 162806 MCS/sec
  beta                 = 2
  Total Energy         = -1.79425 +- 0.00276793
  Average sign         = 0.44562 +- 0.000427331
  Energy per site      = -0.179425 +- 0.000276793
  Specific heat        = 0.192348 +- 0.00399936
  magnetization        = -0.000267411 +- 0.000328735
  susceptibility       = 1.18581 +- 0.0015155
  G                    = 5.00911 +- 0.00418627
  */
}

TEST(WormSimuObs, HXXX1D_2S_NS)
{

  model::base_lattice lat = chain(5);
  vector<size_t> dofs = {4};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.1;
  bool zw = true;

  std::string ham_path, obs_path, wobs_path;

  // This hamiltonian requires zero worm.
  ham_path = "../gtest/model_array/Heisenberg/1D/2sites/Jz_-1_Jx_0.5_Jy_0.3_hz_0_hx_0.5/H";
  obs_path = "../gtest/model_array/Heisenberg/1D/2sites/Jz_-1_Jx_0.5_Jy_0.3_hz_0_hx_0.5/Sz";
  wobs_path = "../gtest/model_array/worm_obs/g_test_2site";
  model::base_model<MC> spin(lat, dofs,
                             ham_path, params, types, shift, zw, false, false);
  // cerr << heisenberg1D_hams << endl;

  model::observable obs(spin, obs_path, false);
  model::WormObs wobs(spin.sps_sites(0), wobs_path);
  vector<batch_res> res;
  double T;
  size_t sweeps, therms;

  sweeps = 5000000;
  T = 0.5;
  therms = 100000;

  run_worm(spin, T, sweeps, therms, res, obs, lat, wobs, 2);

  /*
  expect the following res

  LL = 5 {'Jz': -1, 'Jx': 0.5, 'Jy': 0.3, 'hz': 0, 'hx': 0.5}
  T               = 0.5
  E               = -0.17920078337192535
  C               = 0.1904521882534027
  M               = 3.671157742246578e-07
  M^2             = 6.03879976272583
  G               = 1.1070225238800049
  chi             = 1.1863488077976516


  N = 5,000,000
  temperature          = 0.5
  Average sign         = 0.444791 +- 0.000659766
  -------------------------------------------------
  Energy per site      = -0.179419 +- 0.000279616
  Specific heat        = 0.190624 +- 0.00291661
  magnetization        = 0.000211262 +- 0.000411468
  susceptibility       = 1.18732 +- 0.00216208
  G                    = 1.10602 +- 0.0093439
  Physical configurations = 0.0324196 +- 0
  */
}

TEST(WormSimuObs, HXXX2D_0)
{

  std::vector<size_t> shapes = {3, 4};
  model::base_lattice lat("square lattice", "simple2d", shapes, "../config/lattices.xml", false);
  vector<size_t> dofs = {2};
  std::vector<double> params = {1.0};
  std::vector<int> types = {0};
  double shift = 0.1;
  bool zw = true;

  std::string ham_path, obs_path, wobs_path, wobs_path2;
  ham_path = "../gtest/model_array/Heisenberg/2D/original/Jz_-1_Jx_0.5_Jy_0.3_hz_0_hx_0.5/H";
  obs_path = "../gtest/model_array/Heisenberg/2D/original/Jz_-1_Jx_0.5_Jy_0.3_hz_0_hx_0.5/Sz";
  wobs_path = "../gtest/model_array/worm_obs/g_test5";
  // wobs_path2 = "../gtest/model_array/worm_obs/g_test_2site_2";

  model::base_model<MC> spin(lat, dofs,
                             ham_path, params, types, shift, zw, false, false);

  model::observable obs(spin, obs_path, false);
  model::WormObs wobs(spin.sps_sites(0), wobs_path);
  // model::WormObs wobs2(spin.sps_sites(0), wobs_path2);
  // model::MapWormObs map_wobs(make_pair("G",wobs), make_pair("G2",wobs2));
  model::MapWormObs map_wobs(make_pair("G", wobs));

  vector<batch_res> res;
  double T;
  size_t sweeps, therms;

  sweeps = 5000000;
  T = 0.5;
  therms = 100000;

  run_worm(spin, T, sweeps, therms, res, obs, lat, map_wobs);

  /*
  expect the following res

  L = 12 {'Jz': -1, 'Jx': 0.5, 'Jy': 0.3, 'hz': 0, 'hx': 0.5}
  T               = 0.5
  E               = -0.4478933811187744
  C               = 0.5885588526725769
  M               = -3.6841049677605042e-06
  M^2             = 28.596725463867188
  G               = 0.5638363361358643
  G2              = 6.6042680740356445

  N = 5,000,000
  Elapsed time         = 7.978(7.978) sec
  Speed                = 137879 MCS/sec
  beta                 = 2
  Total Energy         = -5.39727 +- 0.0130383
  Average sign         = 0.299774 +- 0.00145004
  Energy per site      = -0.449772 +- 0.00108653
  Specific heat        = 0.597761 +- 0.0159478
  magnetization        = 0.00226748 +- 0.00324176
  susceptibility       = 4.75886 +- 0.0135557
  G                    = 6.60835 +- 0.0233231
  */
}
