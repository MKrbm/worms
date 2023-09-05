#include <automodel.hpp>
#include <vector>
#include <string>

#include <funcs.hpp>
#include <state.hpp>
#include <automodel.hpp>
#include <autoobservable.hpp>
#include <exec_parallel.hpp>
#include <options.hpp>
#include <argparse.hpp>
#include <observable.hpp>
#include <funcs.hpp>

#include <jackknife.hpp>
#include "gtest/gtest.h"
#include "dataset.hpp"

using namespace std;

struct mc_res {
  struct res {
    double mean;
    double err;
  };
  res ene; // energy per site
  res as;  // average sign
};

typedef bcl::st2013 MC;
typedef std::exponential_distribution<> expdist_t;
typedef std::uniform_real_distribution<> uniform_t;
typedef spin_state::state_t state_t;
typedef spin_state::Operator OP_type;

uniform_t uniform;
// int seed = static_cast<unsigned>(time(0));
int seed = 1681255693;
auto rand_src = engine_type(seed);

std::vector<size_t> shapes = {4};
model::base_lattice lat("chain lattice", "simple1d", shapes, "../config/lattices.xml", false);
string ham_path = "/home/user/project/python/rmsKit/array/HXYZ/original/none/Jx_1_Jy_1_Jz_1_hx_0_hz_0/H";
double alpha = 0.9;
double shift = 0.1;
model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false, true, alpha);
size_t sps = spin.sps_sites(0);
spin_state::StateFunc state_func(sps);
spin_state::StateFunc bond_func(sps * sps);

TEST(HXXX, SplitTest)
{
  auto lops = spin.loperators;

  for (int i = 0; i < lops.size(); i++)
  {
    auto lop = lops[i];
    int size = lop.size;
    int N = size * size;
    for (int j = 0; j < N; j++)
    {
      auto index = state_func.num2state(j, 4);
      auto bond_index = bond_func.num2state(j, 2);
      if (bond_index[0] == bond_index[1])
      {
        double val = lop.ham_vector(j);
        val += lop.single_flip(0, index[1])[index[0]][index[0]] + lop.single_flip(1, index[0])[index[1]][index[1]];
        EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
      }
      else
      {
        if (index[0] == index[2])
        {
          double val = lop.single_flip(true, index[0])[index[1]][index[3]];
          EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
          EXPECT_FLOAT_EQ(val, 0);
        }
        else if (index[1] == index[3])
        {
          double val = lop.single_flip(false, index[1])[index[0]][index[2]];
          EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
          EXPECT_FLOAT_EQ(val, 0);
        }
        else
        {
          EXPECT_NEAR(lop.ham_vector(j) * lop.signs[j], lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
        }
      }
    }
  }
}

TEST(HXXX, WormUpdate)
{
  double beta = 1;
  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  if (wobs_paths.size() == 0)
    wobs_paths.push_back("");
  for (int i = 0; i < wobs_paths.size(); i++)
  {
    string name = "G";
    name += to_string(i);
    mapwobs.push_back(name, model::WormObs(spin.sps_sites(0), wobs_paths[i], 0));
  }

  Worm<MC> solver(beta, spin, mapwobs, cutoff_l, 0); // template needs for std=14

  for (int n = 0; n < 100; n++)
  {

    solver.initStates();
    state_t state = solver.state;
    double sum = 0;
    auto lops = spin.loperators;
    EXPECT_EQ(lops.size(), 1);
    auto lop = lops[0];
    for (auto b : solver.bonds)
    {
      int x1, x2;
      x1 = state[b[0]];
      x2 = state[b[1]];
      int s = x1 + sps * x2;
      sum += lop.ham_prime()[s][s];
    }

    double sum_prime = 0;
    for (auto b : solver.bonds)
    {
      int x1, x2;
      x1 = state[b[0]];
      x2 = state[b[1]];
      int s = x1 + sps * x2;
      sum_prime += lop.ham_vector(s + (sps * sps) * s);
    }

    for (int i = 0; i < solver.L; i++)
    {
      int x1;
      x1 = state[i];
      sum_prime += solver.get_single_flip_elem(i, x1, x1, state);
    }
    EXPECT_NEAR(sum, sum_prime, 1E-8);
  }
  double wdensity = 5;
  double wlength = 0;
  double wcount = 0;
  auto lop = spin.loperators[0];
  auto ham_vector = lop.ham_vector();
  vector<double> markov_mat_elem(4 * sps, 0);
  int t_bond = 0;
  double sum = 0;
  // int fl0 = static_cast<int>(uniform(rand_src) * (sps-1))+1;
  // int leg_index0 = static_cast<int>(3*uniform(rand_src));
  int fl0 = 1;
  int leg_index0 = 0;
  auto bonds = solver.bonds;
  const auto& bond = bonds[t_bond];
  state_t state0 = {0, 1, 0};
  size_t u = state_func.state2num(state0, bond);
  for (int j=1; j < 4*sps; j++){
    OP_type op = OP_type(&bond, &solver.pows_vec[0], u + 4 * u, 0, 0);
    op.update_state(leg_index0, (sps - fl0)%sps);
    int leg_index = j / sps;
    int fl = j % sps;
    int num = op.update_state(leg_index, fl);
    markov_mat_elem[j] = lop.ham_vector(num);
    if (fl == 0) {
      markov_mat_elem[j] = 0;
    }
    sum += markov_mat_elem[j];
  }
  int fl = 1;
  int leg_index = leg_index0;
  OP_type op = OP_type(&bond, &solver.pows_vec[0], u + 4 * u, 0, 0);
  vector<double> markov_cnt(4 * sps, 0);
  int N = 1E5;
  for (int n=0; n < N; n++){
    fl = (sps - fl)%sps;
    int num = op.update_state(leg_index, fl);
    int state_num = op.state();
    auto tmp = lop.markov[state_num](leg_index * (sps - 1) + sps - fl, rand_src);
    if (tmp != 0) {
      tmp --;
      leg_index = tmp;
      int num_prime = op.update_state(leg_index, 1);
      markov_cnt[fl + (sps) * leg_index] ++;
    } else {
      throw std::runtime_error("tmp is zero");
    }
  }

  for (int j=0; j < 4 * sps; j++){
    EXPECT_NEAR(markov_cnt[j] / N, markov_mat_elem[j] / sum, 1E-2);
  }
  
  // for (int i = 0; i < 1E4; i++)
  // {
  //   solver.diagonalUpdate(wdensity);
  //   solver.wormUpdate(wcount, wlength);
  //   for (auto op : solver.ops_main)
  //   {
  //     int off_diag_count = 0;
  //     if (op.is_off_diagonal()){
  //       state_t lstate = op.get_state_vec();
  //       off_diag_count++;
  //     }
  //     if (op._check_is_bond()){
  //       cerr << "off diag count: " << off_diag_count << " elem:" << lop.ham_vector(op.state()) << endl;
  //       if (off_diag_count == 0){
  //         cerr << "count is zero" << endl;
  //       }
  //     }
  //   }
  // }

}



TEST(HXXX, DiagonalUpdate)
{
  double beta = 1;
  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  if (wobs_paths.size() == 0)
    wobs_paths.push_back("");
  for (int i = 0; i < wobs_paths.size(); i++)
  {
    string name = "G";
    name += to_string(i);
    mapwobs.push_back(name, model::WormObs(sps, wobs_paths[i], 0));
  }

  Worm<MC> solver(beta, spin, mapwobs, cutoff_l, 0); // template needs for std=14
  for (int n = 0; n < 8; n++){
    solver.initStates();
    auto state_ = state_func.num2state(n, 3);
    solver.state[0] = state_[0];
    solver.state[1] = state_[1];
    solver.state[2] = state_[2];


    double wdensity = 5;

    vector<vector<size_t>> &bonds = solver.bonds;
    vector<vector<size_t>> &sites = solver.nn_sites;

    auto state = solver.state;
    vector<double> weights_bonds(bonds.size(), 0);
    vector<double> weights_sites(sites.size(), 0);
    for (int i = 0; i < bonds.size(); i++)
    {
      double bop_label = solver.bond_type[i];
      auto const &accept = solver.accepts[bop_label];
      auto const &bond = solver.bonds[i];
      size_t u = solver.state_funcs[bop_label].state2num(state, bond);
      weights_bonds[i] = accept[u] * solver.max_diagonal_weight;
    }

    for (int i = 0; i < sites.size(); i++)
    {
      //* append single-flip
      double sop_label = spin.site_type[i];
      double mat_elem = 0;
      for (auto target : spin.nn_sites[i])
      {
        mat_elem += spin.loperators[target.bt]
                        .single_flip(target.start, state[target.target], state[i], state[i]);
      }

      mat_elem = std::abs(mat_elem);
      weights_sites[i] = mat_elem;
    }

    std::vector<BC::observable> bops(spin.Nb);
    std::vector<BC::observable> sops(spin.L);

    for (int i=0; i<1E5; i++) {
      std::vector<double> bop(bonds.size(), 0);
      std::vector<double> sop(sites.size(), 0);
      solver.diagonalUpdate(wdensity);
      for (auto op : solver.ops_main)
      {
        if (op.op_type() >= 0)
        {
          ptrdiff_t index = op.bond_ptr() - &bonds[0];
          bop[index] += 1;
        }
        else
        {
          ptrdiff_t index = op.bond_ptr() - &sites[0];
          sop[index] += 1;
        }
      }
      for (int b = 0; b < bonds.size(); b++)
      {
        bops[b] << bop[b];
      }
      for (int s = 0; s < sites.size(); s++)
      {
        sops[s] << sop[s];
      }
    }

    for (int b = 0; b < bonds.size(); b++){
      EXPECT_NEAR(bops[b].mean(), weights_bonds[b], 5 * bops[b].error());
    }

    for (int s = 0; s < sites.size(); s++){
      EXPECT_NEAR(sops[s].mean(), weights_sites[s], 5 * sops[s].error());
    }
  }
}



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



TEST(HXX1D_02a, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.2;
  double shift = 0.1;
  std::vector<size_t> shapes = {4};
  model::base_lattice lat("chain lattice", "simple1d", shapes, "../config/lattices.xml", false);
  string ham_path = "/home/user/project/python/rmsKit/array/HXYZ/original/none/Jx_1_Jy_1_Jz_1_hx_0_hz_0/H";
  model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false, false, alpha);
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
  EXPECT_NEAR(out_res.ene.mean, -0.21627057785439316,
              3 * out_res.ene.err); // -0.21627057785439316 for L = 4 J = [1, 1, 1]
}
