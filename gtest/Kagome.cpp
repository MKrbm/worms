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



typedef bcl::st2013 MC;
typedef std::exponential_distribution<> expdist_t;
typedef std::uniform_real_distribution<> uniform_t;
typedef spin_state::state_t state_t;
typedef spin_state::Operator OP_type;

uniform_t uniform;
// int seed = static_cast<unsigned>(time(0));
int seed = 1681255693;
auto rand_src = engine_type(seed);

std::vector<size_t> shapes = {4, 4};
model::base_lattice lat("triangular lattice", "anisotropic triangular", shapes, "../config/lattices.xml", false);
// string ham_path = "../gtest/model_array/KH/smel/H1";
string ham_path = "/home/user/project/python/rmsKit/array/KH/3site/none/Jx_1_Jy_1_Jz_1_hx_0_hz_0/H";
model::base_model<MC> spin(lat, {8}, ham_path, {1, 1, 1}, {0, 1, 2}, 0.1, false, false, true);

model::base_lattice lat2("triangular lattice", "kagome", shapes, "../config/lattices.xml", false);
string ham_path2 = "/home/user/project/python/rmsKit/array/KH/original/none/Jx_1_Jy_1_Jz_1_hx_0_hz_0/H";
model::base_model<MC> spin2(lat2, {2}, ham_path2, {1}, {0}, 0.3, false, false, true);

void run_worm(
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
       << endl;
  return;
}

TEST(ModelTest, Kagome)
{

  double T = 2;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 1000;
  therms = 100;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;
  
  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);

  run_worm(spin2, T, sweeps, therms, res, obs, lat2, mapwobs);
}
