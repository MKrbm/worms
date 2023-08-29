#include <vector>
#include <string>
#include <funcs.hpp>
#include <alps/alea/batch.hpp>
#include <alps/utilities/mpi.hpp>
#include <alps/alea/mpi.hpp>

#include "exec_parallel.hpp"
#include <automodel.hpp>
#include <jackknife.hpp>

#include "gtest/gtest.h"
#include "dataset.hpp"

using namespace std;

typedef bcl::st2013 MC;
typedef std::exponential_distribution<> expdist_t;
typedef std::uniform_real_distribution<> uniform_t;
typedef spin_state::state_t state_t;
typedef spin_state::Operator OP_type;


// This is a simplest test (1D Ising model)

uniform_t uniform;

int seed = 1681255693;
auto rand_src = engine_type(seed);

std::vector<size_t> shapes = {2};
model::base_lattice lat("chain lattice", "simple1d", shapes, "../config/lattices.xml", false);
string ham_path = "/home/user/project/gtest/model_array/Ising1D/test1";
double alpha = 0;
double shift = 0.1;
model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false, true, alpha);
size_t sps = spin.sps_sites(0);
spin_state::StateFunc state_func(sps);
spin_state::StateFunc bond_func(sps * sps);


struct mc_res
{
  struct res
  {
    double mean;
    double err; 
  };

  res ene; // energy per site
  res as; // average sign
};

mc_res run_worm(
    model::base_model<MC> &spin,
    double T, size_t sweeps, size_t therms,
    std::vector<batch_res> &res,
    model::observable &obs,
    model::base_lattice &lat,
    model::MapWormObs wobs)
{
  // dont fix worm density. Not printout density information.
  alps::alea::autocorr_result<double> ac_res;
  exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, ac_res, obs, wobs);

  batch_res as = res[0];  // average sign
  batch_res ene = res[1]; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res sglt = res[2];
  batch_res n_neg_ele = res[3];
  batch_res n_ops = res[4];
  batch_res N2 = res[5];
  batch_res N = res[6];

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



  // cout << lat.L << endl;
  // cout << "temperature          = " << T
  //      << endl
  //      << "Average sign         = "
  //      << as_mean.first << " +- "
  //      << as_mean.second
  //      << endl;
  // cout << "-------------------------------------------------" << endl;
  // cout << "Energy per site      = "
  //      << ene_mean.first / lat.L << " +- "
  //      << ene_mean.second / lat.L
  //      << endl
  //      << "Specific heat        = "
  //      << c_mean.first / lat.L << " +- "
  //      << c_mean.second / lat.L
  //      << endl;
  mc_res res_;
  res_.ene = {ene_mean.first / lat.L, ene_mean.second / lat.L};
  res_.as = {as_mean.first, as_mean.second};
  return res_;
}

// TEST(Ising1D, MC)
// {
//   double T = 2;
//   double beta = 1 / T;
//   size_t sweeps, therms;
//   sweeps = 1000;
//   therms = 100;
//
//   size_t cutoff_l = std::numeric_limits<size_t>::max();
//   model::MapWormObs mapwobs;
//   
//   vector<string> wobs_paths;
//   wobs_paths.push_back("");
//   std::vector<batch_res> res;
//   model::observable obs(spin, "", false);
//
//   //run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);
//   
//   mc_res res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);
//   
//   EXPECT_NEAR(res.ene.mean, "TBD" 0.0001);
//   EXPECT_NEAR(res.ene.err, "TBD" 0.0001);
// }
