#include <argparse.hpp>
#include <automodel.hpp>
#include <autoobservable.hpp>
#include <exec_parallel.hpp>
#include <funcs.hpp>
#include <jackknife.hpp>
#include <observable.hpp>
#include <string>
#include <utility>
#include <vector>

#include "dataset.hpp"
#include "gtest/gtest.h"
#define SEED 16625036


struct mc_res {
  struct res {
    std::complex<double> mean;
    std::complex<double> err;
  };
  res ene;  // energy per site
  res as;   // average sign
};

using MC = bcl::st2013;
using expdist_t = std::exponential_distribution<>;
using uniform_t = std::uniform_real_distribution<>;
using state_t = spin_state::state_t;
using OP_type = spin_state::Operator;


uniform_t uniform;
// int seed = static_cast<unsigned>(time(0));
int seed = 1681255693;
auto rand_src = engine_type(SEED);

mc_res run_worm(model::base_model<MC> &spin, double T, size_t sweeps,
                size_t therms, std::vector<batch_res_complex> &res,
                model::observable &obs, model::base_lattice &lat,
                model::MapWormObs wobs) {
  // dont fix worm density. Not printout density information.
  alps::alea::autocorr_result<double> ac_res;
  double r;
  exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, ac_res, obs,
                    std::move(wobs), r, SEED);

  batch_res_complex as = res[0];   // average sign
  batch_res_complex ene = res[1];  // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res_complex sglt = res[2];
  batch_res_complex n_neg_ele = res[3];
  batch_res_complex n_ops = res[4];
  batch_res_complex N2 = res[5];
  batch_res_complex N = res[6];

  std::function<std::complex<double>(std::complex<double>,std::complex<double>,std::complex<double>)> f;

  pair<std::complex<double>, std::complex<double>> as_mean =
      jackknife_reweight_single(as);  // calculate <S>
  pair<std::complex<double>, std::complex<double>> nop_mean =
      jackknife_reweight_single(n_ops);  // calculate <S>
  pair<std::complex<double>, std::complex<double>> nnop_mean =
      jackknife_reweight_single(n_neg_ele);  // calculate <S>

  // calculate energy
  pair<std::complex<double>, std::complex<double>> ene_mean =
      jackknife_reweight_div(ene, as);  // calculate <SH> / <S>

  // calculat heat capacity
  f = [](std::complex<double> x1, std::complex<double> x2, std::complex<double> y) {
    return (x2 - x1) / y - (x1 / y) * (x1 / y);
  };
  pair<std::complex<double>, std::complex<double>> c_mean = jackknife_reweight_any(N, N2, as, f);

  mc_res res_;
  res_.ene = {ene_mean.first / (double)lat.L, ene_mean.second / (double)lat.L};
  res_.as = {as_mean.first, as_mean.second};
  return res_;
}

TEST(BLBQ1D_a, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.1;
  double shift = 0.1;
  std::vector<size_t> shapes = {6};
  // PBC
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                          "../config/lattices.xml", false);
  string u_path = "../gtest/model_array/blbq1d/analytical/0.npy";
  string ham_path = "../gtest/model_array/blbq1d/J0_1_J1_-0.1_hx_0_hz_0/1_mel/H";
  model::base_model<MC> spin(lat, {3}, ham_path, {1}, {0}, shift, true, false,
                             false, alpha);
  double T = 1;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 1000000;
  therms = 10000;

  size_t cutoff_l = 200;
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res_complex> res;
  model::observable obs(spin, "", false);

  // run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  // EXPECT_FLOAT_EQ(out_res.ene.mean, -0.21661999999999898);
  EXPECT_NEAR(
      std::real(out_res.ene.mean), -1.3652522251325874,
      3 * std::real(out_res.ene.err));  // -0.21627057785439316 for L = 4 J = [1, 1, 1]

  std::cerr << "energy mean: " << out_res.ene.mean << " ± " << out_res.ene.err << std::endl;
  std::cerr << "as mean: " << out_res.as.mean << " ± " << out_res.as.err << std::endl;
}

TEST(BLBQ1D_b_1, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.1;
  double shift = 0.1;
  std::vector<size_t> shapes = {6};
  // PBC
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                          "../config/lattices.xml", false);
  string ham_path = "../gtest/model_array/blbq1d/J0_1_J1_-0.1_hx_0.5_hz_0/1_mel/H";
  model::base_model<MC> spin(lat, {3}, ham_path, {1}, {0}, shift, true, false,
                             false, alpha);
  double T = 1;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 1000000;
  therms = 10000;

  size_t cutoff_l = 200;
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res_complex> res;
  model::observable obs(spin, "", false);

  // run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  // EXPECT_FLOAT_EQ(out_res.ene.mean, -0.21661999999999898);
  EXPECT_NEAR(
      std::real(out_res.ene.mean), -1.377080758813392,
      3 * std::real(out_res.ene.err));  // -0.21627057785439316 for L = 4 J = [1, 1, 1]
  
  std::cerr << "energy mean: " << out_res.ene.mean << " ± " << out_res.ene.err << std::endl;
  std::cerr << "as mean: " << out_res.as.mean << " ± " << out_res.as.err << std::endl;
}

TEST(BLBQ1D_b_2, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.1;
  double shift = 0.1;
  std::vector<size_t> shapes = {6};
  // PBC
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                          "../config/lattices.xml", false);
  string u_path = "../gtest/model_array/blbq1d/analytical";
  string ham_path = "../gtest/model_array/blbq1d/J0_1_J1_-0.1_hx_0.5_hz_0/1_mel/H";
  model::base_model<MC> spin(lat, {3}, ham_path, u_path, {1}, {0}, shift, true, false,
                             false, alpha);
  double T = 1;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 1000000;
  therms = 10000;

  size_t cutoff_l = 200;
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res_complex> res;
  model::observable obs(spin, "", false);

  // run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  // EXPECT_FLOAT_EQ(out_res.ene.mean, -0.21661999999999898);
  EXPECT_NEAR(
      std::real(out_res.ene.mean), -1.377080758813392,
      3 * std::real(out_res.ene.err));  
    
  EXPECT_NEAR(
      std::imag(out_res.as.mean), 0,
      3 * std::imag(out_res.as.err));  
  
  std::cerr << "energy mean: " << out_res.ene.mean << " ± " << out_res.ene.err << std::endl;
  std::cerr << "as mean: " << out_res.as.mean << " ± " << out_res.as.err << std::endl;
}

TEST(BLBQ1D_c, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.1;
  double shift = 0.1;
  std::vector<size_t> shapes = {6};
  // PBC
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                          "../config/lattices.xml", false);
  string u_path = "../gtest/model_array/blbq1d/u_complex";
  string ham_path = "../gtest/model_array/blbq1d/J0_1_J1_-0.5_hx_0.3_hz_0/1_mel/H";
  model::base_model<MC> spin(lat, {3}, ham_path, u_path, {1}, {0}, shift, true, false,
                             false, alpha);
  double T = 1;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 1000000;
  therms = 10000;

  size_t cutoff_l = 200;
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res_complex> res;
  model::observable obs(spin, "", false);

  // run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  // EXPECT_FLOAT_EQ(out_res.ene.mean, -0.21661999999999898);
  EXPECT_NEAR(
      std::real(out_res.ene.mean), -2.528433141529662,
      3 * std::real(out_res.ene.err));  
    
  EXPECT_NEAR(
      std::imag(out_res.as.mean), 0,
      3 * std::imag(out_res.as.err));  

  EXPECT_NEAR(
      std::real(out_res.as.mean), 0.5903428752205623,
      3 * std::real(out_res.as.err));  
  
  std::cerr << "energy mean: " << out_res.ene.mean << " ± " << out_res.ene.err << std::endl;
  std::cerr << "as mean: " << out_res.as.mean << " ± " << out_res.as.err << std::endl;
}