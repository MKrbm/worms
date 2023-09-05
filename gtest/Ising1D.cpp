#include <alps/alea/batch.hpp>
#include <alps/alea/mpi.hpp>
#include <alps/utilities/mpi.hpp>
#include <funcs.hpp>
#include <string>
#include <vector>

#include "exec_parallel.hpp"
#include <automodel.hpp>
#include <jackknife.hpp>

#include "dataset.hpp"
#include "gtest/gtest.h"

using namespace std;

typedef bcl::st2013 MC;
typedef std::exponential_distribution<> expdist_t;
typedef std::uniform_real_distribution<> uniform_t;
typedef spin_state::state_t state_t;
typedef spin_state::Operator OP_type;

// This is a simplest test (1D Ising model)

// This return energy per site for 1D Ising model with J = 1 and L = 2.
// 1/4.0 since local hamiltonian is overwrapped for tow sites.
double energy_ising_2(double beta) {
  return -1.0/4 * tanh(beta / 2.0);
}

double n_diag_ising_2(double beta, double shift, int L = 2) {
  return  L * 1 * beta * (0.5 + shift);
}

double n_diag_ising(double beta, double shift, int L = 2) {
  return L * 1 * beta * (shift);
}

double n_diag_all(double beta, double shift){
  double h0 = 2*(0.5 + shift);
  double h1 = 2*shift;
  return beta * (h0 * exp(h0*beta) + h1 * exp(h1*beta)) / (exp(h0*beta) + exp(h1*beta));
}

uniform_t uniform;

int seed = 1681255693;
auto rand_src = engine_type(seed);

std::vector<size_t> shapes = {2};
model::base_lattice lat("chain lattice", "simple1d", shapes,
                        "../config/lattices.xml", false);
string ham_path = "/home/user/project/gtest/model_array/Ising1D/test1";
double alpha = 0;
double shift = 0.5;
model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false,
                           false, alpha);
size_t sps = spin.sps_sites(0);
spin_state::StateFunc state_func(sps);
spin_state::StateFunc bond_func(sps *sps);

struct mc_res {
  struct res {
    double mean;
    double err;
  };

  res ene; // energy per site
  res as;  // average sign
};

mc_res run_worm(model::base_model<MC> &spin, double T, size_t sweeps,
                size_t therms, std::vector<batch_res> &res,
                model::observable &obs, model::base_lattice &lat,
                model::MapWormObs wobs) {
  // dont fix worm density. Not printout density information.
  alps::alea::autocorr_result<double> ac_res;
  exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, ac_res, obs,
                    wobs);

  batch_res as = res[0];  // average sign
  batch_res ene = res[1]; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res sglt = res[2];
  batch_res n_neg_ele = res[3];
  batch_res n_ops = res[4];
  batch_res N2 = res[5];
  batch_res N = res[6];

  std::function<double(double, double, double)> f;

  pair<double, double> as_mean = jackknife_reweight_single(as); // calculate <S>
  pair<double, double> nop_mean =
      jackknife_reweight_single(n_ops); // calculate <S>
  pair<double, double> nnop_mean =
      jackknife_reweight_single(n_neg_ele); // calculate <S>

  // calculate energy
  pair<double, double> ene_mean =
      jackknife_reweight_div(ene, as); // calculate <SH> / <S>

  // calculat heat capacity
  f = [](double x1, double x2, double y) {
    return (x2 - x1) / y - (x1 / y) * (x1 / y);
  };
  pair<double, double> c_mean = jackknife_reweight_any(N, N2, as, f);

  mc_res res_;
  res_.ene = {ene_mean.first / lat.L, ene_mean.second / lat.L};
  res_.as = {as_mean.first, as_mean.second};
  return res_;
}

TEST(DiagonalUpdate, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 1;
  double beta = 1;
  std::vector<size_t> shapes = {4};
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                        "../config/lattices.xml", false);
  model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false,
                             false, alpha);
  Worm<MC> solver(beta, spin, {}, -1, 0);
  BC::observable op_cnt;

  int sweeps = 1000000;
 
  for (int i = 0; i < sweeps; i++) {
    solver.diagonalUpdate(1);
    op_cnt << solver.ops_main.size();
    solver.ops_main.resize(0);
  }
  EXPECT_NEAR(op_cnt.mean(), n_diag_ising_2(beta, shift, shapes[0]),
              3 * op_cnt.error()); 

  // for different initial state
  BC::observable op_cnt2;

  // initialize state with antifero
  for (int i = 0; i < solver.state.size(); i++) {
    solver.state[i] = i % 2;
  }
  for (int i = 0; i < sweeps; i++) {
    solver.diagonalUpdate(1);
    op_cnt2 << solver.ops_main.size();
    solver.ops_main.resize(0);
  }

  EXPECT_NEAR(op_cnt2.mean(), n_diag_ising(beta, shift, shapes[0]),
              3 * op_cnt2.error()); 


  // transition through all states
  // BC::observable op_cnt3;
  // double wcount, wlength;
  // for (int i = 0; i < sweeps; i++) {
  //   solver.diagonalUpdate(1);
  //   solver.wormUpdate(wcount, wlength);
  //   // std::cerr << solver.state << std::endl;
  //   op_cnt3 << solver.ops_main.size();
  //   // solver.ops_main.resize(0);
  // }
  // EXPECT_NEAR(op_cnt3.mean(), n_diag_all(beta, shift),
  //             3 * op_cnt3.error());

}

TEST(Ising1D_half_a, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.5;
  std::vector<size_t> shapes = {4};
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                        "../config/lattices.xml", false);
  model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false,
                             false, alpha);
  double T = 1;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 1000000;
  therms = 0;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);

  // run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  // check if result is withing 3 sigma.
  EXPECT_NEAR(out_res.ene.mean, -0.06466984014315179,
              3 * out_res.ene.err); // energy per site
}
TEST(Ising1D_1a, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 1;
  std::vector<size_t> shapes = {4};
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                        "../config/lattices.xml", false);
  model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false,
                             false, alpha);
  double T = 1;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 1000000;
  therms = 0;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);

  // run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  // check if result is withing 3 sigma.
  EXPECT_NEAR(out_res.ene.mean, -0.06466984014315179,
              3 * out_res.ene.err); // energy per site
}

TEST(Ising1D_0a, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0;
  std::vector<size_t> shapes = {4};
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                        "../config/lattices.xml", false);
  model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false,
                             false, alpha);
  double T = 1;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 1000000;
  therms = 0;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);

  // run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  // check if result is withing 3 sigma.
  EXPECT_NEAR(out_res.ene.mean, -0.06466984014315179,
              3 * out_res.ene.err); // energy per site
}
