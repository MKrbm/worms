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
#define SEED 16625035


struct mc_res {
  struct res {
    double mean;
    double err;
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

std::vector<size_t> shapes = {4};
model::base_lattice lat("chain lattice", "simple1d", shapes,
                        "../config/lattices.xml", false);
string ham_path =
    "../gtest/model_array/Heisenberg/1D/original/"
    "Jx_1_Jy_1_Jz_1_hx_0_hz_0/H/";
double alpha = 0.9;
double shift = 0.1;
model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false,
                           false, alpha); // print = false
size_t sps = spin.sps_sites(0);
spin_state::StateFunc state_func(sps);
spin_state::StateFunc bond_func(sps *sps);

mc_res run_worm(model::base_model<MC> &spin, double T, size_t sweeps,
                size_t therms, std::vector<batch_res> &res,
                model::observable &obs, model::base_lattice &lat,
                model::MapWormObs wobs) {
  // dont fix worm density. Not printout density information.
  alps::alea::autocorr_result<double> ac_res;
  double r;
  exe_worm_parallel(spin, T, sweeps, therms, -1, false, true, res, ac_res, obs,
                    std::move(wobs), r, SEED);

  batch_res as = res[0];   // average sign
  batch_res ene = res[1];  // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res sglt = res[2];
  batch_res n_neg_ele = res[3];
  batch_res n_ops = res[4];
  batch_res N2 = res[5];
  batch_res N = res[6];

  std::function<double(double, double, double)> f;

  pair<double, double> as_mean =
      jackknife_reweight_single(as);  // calculate <S>
  pair<double, double> nop_mean =
      jackknife_reweight_single(n_ops);  // calculate <S>
  pair<double, double> nnop_mean =
      jackknife_reweight_single(n_neg_ele);  // calculate <S>

  // calculate energy
  pair<double, double> ene_mean =
      jackknife_reweight_div(ene, as);  // calculate <SH> / <S>

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

TEST(HXX1D2SITE, check_update_a) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.2;
  double shift = 0.1;
  std::vector<size_t> shapes = {2};
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                          "../config/lattices.xml", false);
  string ham_path =
      "../gtest/model_array/Heisenberg/1D/2sites/Jx_-0.3_Jy_0.5_Jz_0.8_hx_0.3_hz_0_lt_2/";
  bool zero_worm = false;
  model::base_model<MC> spin(lat, {4}, ham_path, {1}, {0}, shift, zero_worm,
                             false, false, alpha);

  double T = 0.3;
  double beta = 1 / T;
  size_t sweeps;
  size_t therms;
  sweeps = 500000;
  therms = 0;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.emplace_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);
  auto lop = spin.loperators[0];

  Worm<MC> solver(beta, spin, mapwobs, cutoff_l, 0, SEED);

  typedef spin_state::Operator OP_type;
  typedef std::vector<OP_type> OPS;
  using state_t = spin_state::state_t;

  // n* loop until ops_main has 3 operators
  double wdensity = 5;
  double wcount = 1;
  double wlength = 1;
  size_t w_upd_cnt;
  size_t cutoff_thres = std::numeric_limits<size_t>::max();
  while (true) {
    solver.diagonalUpdate(10);  // n* need to be comment out
    solver.wormUpdate(wcount, wlength, w_upd_cnt, cutoff_thres);
    auto ops = solver.ops_main;
    int cnt = 0;
    for (const auto &op : ops) cnt += op.is_diagonal() ? 0 : 1;
    if (ops.size() == 4) break;
  }

  std::vector<state_t> state_list;
  std::map<size_t, size_t> cnt;
  std::map<size_t, double> mat_elem;

  for (int i = 0; i < sweeps; i++) {
    solver.wormUpdate(wcount, wlength, w_upd_cnt, cutoff_thres);
    state_t state = solver.state;
    state_t cstate = state;
    auto ops = solver.ops_main;
    size_t state_num = 0;
    double bolzman = 1;
    for (auto opi = ops.begin(); opi != ops.end(); opi++) {
      state_num += cstate[0] * 4 + cstate[1];
      state_num *= 16;
      if (opi->op_type() >= 0) {
        bolzman *= lop.ham_vector(opi->state());
      } else {
        bolzman *= solver.get_single_flip_elem(*opi);
      }
      solver.update_state(opi, cstate);
    }
    if (cnt.find(state_num) == cnt.end()) {
      cnt[state_num] = 1;
      mat_elem[state_num] = bolzman;
    } else {
      cnt[state_num]++;
    }
  }

  // loop over all key and value of cnt
  double sum_mat = 0;
  double sum_cnt = 0;
  for (auto const &x : cnt) {
    size_t state_num = x.first;
    size_t count = x.second;
    double mat_elem_ = mat_elem[state_num];
    sum_mat += mat_elem_;
    sum_cnt += count;
    // EXPECT_NEAR(count / sweeps, mat_elem_ / solver.sum, 1E-2);
  }

  EXPECT_FLOAT_EQ((double)sweeps, sum_cnt);

  for (auto const &x : cnt) {
    size_t state_num = x.first;
    size_t count = x.second;
    double p = mat_elem[state_num] / sum_mat;
    double p_prime = count / sum_cnt;
    double var = p * (1 - p) / sweeps;
    EXPECT_NE(p, 0);
    if (count != 1) {
      EXPECT_NEAR(p_prime, p, 8 * sqrt(var));
    } else {
      // EXPECT_NEAR(p_prime, p, 20 * sqrt(var));
    }
  }
}

TEST(HXX1D2SITE, check_update) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0;
  double shift = 0.1;
  std::vector<size_t> shapes = {2};
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                          "../config/lattices.xml", false);
  string ham_path =
      "../gtest/model_array/Heisenberg/1D/original/mes/Jx_-0.3_Jy_0.5_Jz_0.8_hx_0.3_hz_0/";
  bool zero_worm = false;
  model::base_model<MC> spin(lat, {4}, ham_path, {1}, {0}, shift, zero_worm,
                             false, false, alpha);

  double T = 0.3;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 500000;
  therms = 0;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);
  auto lop = spin.loperators[0];

  Worm<MC> solver(beta, spin, mapwobs, cutoff_l, 0, SEED);

  typedef spin_state::Operator OP_type;
  typedef std::vector<OP_type> OPS;
  using state_t = spin_state::state_t;

  // n* loop until ops_main has 3 operators
  double wdensity = 5;
  double wcount = 1;
  double wlength = 1;
  size_t w_upd_cnt;
  size_t cutoff_thres = std::numeric_limits<size_t>::max();

  while (true) {
    solver.diagonalUpdate(10);  // n* need to be comment out
    solver.wormUpdate(wcount, wlength, w_upd_cnt, cutoff_thres);
    auto ops = solver.ops_main;
    int cnt = 0;
    for (const auto &op : ops) cnt += op.is_diagonal() ? 0 : 1;
    if (ops.size() == 4) break;
  }

  std::vector<state_t> state_list;
  std::map<size_t, size_t> cnt;
  std::map<size_t, double> mat_elem;

  for (int i = 0; i < sweeps; i++) {
    solver.wormUpdate(wcount, wlength, w_upd_cnt, cutoff_thres);
    state_t state = solver.state;
    state_t cstate = state;
    auto ops = solver.ops_main;
    size_t state_num = 0;
    double bolzman = 1;
    for (typename OPS::iterator opi = ops.begin(); opi != ops.end(); opi++) {
      state_num += cstate[0] * 4 + cstate[1];
      state_num *= 16;
      bolzman *= lop.ham_vector(opi->state());
      solver.update_state(opi, cstate);
    }
    if (cnt.find(state_num) == cnt.end()) {
      cnt[state_num] = 1;
      mat_elem[state_num] = bolzman;
    } else {
      cnt[state_num]++;
    }
  }

  // loop over all key and value of cnt
  double sum_mat = 0;
  double sum_cnt = 0;
  for (auto const &x : cnt) {
    size_t state_num = x.first;
    size_t count = x.second;
    double mat_elem_ = mat_elem[state_num];
    sum_mat += mat_elem_;
    sum_cnt += count;
    // EXPECT_NEAR(count / sweeps, mat_elem_ / solver.sum, 1E-2);
  }

  EXPECT_FLOAT_EQ((double)sweeps, sum_cnt);

  for (auto const &x : cnt) {
    size_t state_num = x.first;
    size_t count = x.second;
    double p = mat_elem[state_num] / sum_mat;
    double p_prime = count / sum_cnt;
    double var = p * (1 - p) / sweeps;
    EXPECT_NE(p, 0);
    if (count != 1) {
      EXPECT_NEAR(p_prime, p, 8 * sqrt(var));
    } else {
      // EXPECT_NEAR(p_prime, p, 20 * sqrt(var));
    }
  }
}

TEST(HXXX, SplitTest) {
  auto lops = spin.loperators;

  for (int i = 0; i < lops.size(); i++) {
    auto lop = lops[i];
    int size = lop.size;
    int N = size * size;
    for (int j = 0; j < N; j++) {
      auto index = state_func.num2state(j, 4);
      auto bond_index = bond_func.num2state(j, 2);
      if (bond_index[0] == bond_index[1]) {
        double val = lop.ham_vector(j);
        val += lop.single_flip(0, index[1])[index[0]][index[0]] +
               lop.single_flip(1, index[0])[index[1]][index[1]];
        EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
      } else {
        if (index[0] == index[2]) {
          double val = lop.single_flip(true, index[0])[index[1]][index[3]];
          EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
          EXPECT_FLOAT_EQ(val, 0);
        } else if (index[1] == index[3]) {
          double val = lop.single_flip(false, index[1])[index[0]][index[2]];
          EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
          EXPECT_FLOAT_EQ(val, 0);
        } else {
          EXPECT_NEAR(lop.ham_vector(j) * lop.signs[j],
                      lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
        }
      }
    }
  }
}

TEST(HXXX, WormUpdate) {
  double beta = 1;
  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  if (wobs_paths.empty()) wobs_paths.emplace_back("");
  for (int i = 0; i < wobs_paths.size(); i++) {
    string name = "G";
    name += to_string(i);
    mapwobs.push_back(name,
                      model::WormObs(spin.sps_sites(0), wobs_paths[i], false));
  }

  Worm<MC> solver(beta, spin, mapwobs, cutoff_l,
                  0);  // template needs for std=14

  for (int n = 0; n < 100; n++) {
    solver.initStates();
    state_t state = solver.state;
    double sum = 0;
    auto lops = spin.loperators;
    EXPECT_EQ(lops.size(), 1);
    auto lop = lops[0];
    for (auto b : solver.bonds) {
      int x1;
      int x2;
      x1 = state[b[0]];
      x2 = state[b[1]];
      int s = x1 + sps * x2;
      sum += lop.ham_prime()[s][s];
    }

    double sum_prime = 0;
    for (auto b : solver.bonds) {
      int x1;
      int x2;
      x1 = state[b[0]];
      x2 = state[b[1]];
      int s = x1 + sps * x2;
      sum_prime += lop.ham_vector(s + (sps * sps) * s);
    }

    for (int i = 0; i < solver.L; i++) {
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
  const auto &bond = bonds[t_bond];
  state_t state0 = {0, 1, 0};
  size_t u = state_func.state2num(state0, bond);
  for (int j = 1; j < 4 * sps; j++) {
    OP_type op = OP_type(&bond, solver.pows_vec.data(), u + 4 * u, 0, 0);
    op.update_state(leg_index0, (sps - fl0) % sps);
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
  OP_type op = OP_type(&bond, solver.pows_vec.data(), u + 4 * u, 0, 0);
  vector<double> markov_cnt(4 * sps, 0);
  int N = 1E5;
  for (int n = 0; n < N; n++) {
    fl = (sps - fl) % sps;
    int num = op.update_state(leg_index, fl);
    int state_num = op.state();
    auto tmp =
        lop.markov[state_num](leg_index * (sps - 1) + sps - fl, rand_src);
    if (tmp != 0) {
      tmp--;
      leg_index = tmp;
      int num_prime = op.update_state(leg_index, 1);
      markov_cnt[fl + (sps)*leg_index]++;
    } else {
      throw std::runtime_error("tmp is zero");
    }
  }

  for (int j = 0; j < 4 * sps; j++) {
    EXPECT_NEAR(markov_cnt[j] / N, markov_mat_elem[j] / sum, 1E-2);
  }
}

TEST(HXXX, DiagonalUpdate) {
  double beta = 1;
  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  if (wobs_paths.empty()) wobs_paths.push_back("");
  for (int i = 0; i < wobs_paths.size(); i++) {
    string name = "G";
    name += to_string(i);
    mapwobs.push_back(name, model::WormObs(sps, wobs_paths[i], 0));
  }

  Worm<MC> solver(beta, spin, mapwobs, cutoff_l,
                  0);  // template needs for std=14
  for (int n = 0; n < 8; n++) {
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
    for (int i = 0; i < bonds.size(); i++) {
      double bop_label = solver.bond_type[i];
      auto const &accept = solver.accepts[bop_label];
      auto const &bond = solver.bonds[i];
      size_t u = solver.state_funcs[bop_label].state2num(state, bond);
      weights_bonds[i] = accept[u] * solver.max_diagonal_weight;
    }

    for (int i = 0; i < sites.size(); i++) {
      //* append single-flip
      double sop_label = spin.site_type[i];
      double mat_elem = 0;
      for (auto target : spin.nn_sites[i]) {
        mat_elem += spin.loperators[target.bt].single_flip(
            target.start, state[target.target], state[i], state[i]);
      }

      mat_elem = std::abs(mat_elem);
      weights_sites[i] = mat_elem;
    }

    std::vector<BC::observable> bops(spin.Nb);
    std::vector<BC::observable> sops(spin.L);

    for (int i = 0; i < 1E5; i++) {
      std::vector<double> bop(bonds.size(), 0);
      std::vector<double> sop(sites.size(), 0);
      solver.diagonalUpdate(wdensity);
      for (auto op : solver.ops_main) {
        if (op.op_type() >= 0) {
          ptrdiff_t index = op.bond_ptr() - &bonds[0];
          bop[index] += 1;
        } else {
          ptrdiff_t index = op.bond_ptr() - &sites[0];
          sop[index] += 1;
        }
      }
      for (int b = 0; b < bonds.size(); b++) {
        bops[b] << bop[b];
      }
      for (int s = 0; s < sites.size(); s++) {
        sops[s] << sop[s];
      }
    }

    for (int b = 0; b < bonds.size(); b++) {
      EXPECT_NEAR(bops[b].mean(), weights_bonds[b], 5 * bops[b].error());
    }

    for (int s = 0; s < sites.size(); s++) {
      EXPECT_NEAR(sops[s].mean(), weights_sites[s], 5 * sops[s].error());
    }
  }
}

TEST(HXX1D_02a, MC) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.2;
  double shift = 0.1;
  std::vector<size_t> shapes = {4};
  model::base_lattice lat("chain lattice", "simple1d", shapes,
                          "../config/lattices.xml", false);
  string ham_path =
      "../gtest/model_array/Heisenberg/1D/"
      "original/Jx_1_Jy_1_Jz_1_hx_0_hz_0/H/";
  model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, false, false,
                             false, alpha);
  double T = 1;
  double beta = 1 / T;
  size_t sweeps, therms;
  sweeps = 400000;
  therms = 0;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.push_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);

  // run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  EXPECT_FLOAT_EQ(out_res.ene.mean, -0.21661999999999898);
  EXPECT_NEAR(
      out_res.ene.mean, -0.21627057785439316,
      3 * out_res.ene.err);  // -0.21627057785439316 for L = 4 J = [1, 1, 1]
}

TEST(HXX2D, none_a_zw) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.2;
  double shift = 0.1;
  bool zero_worm = true;
  std::vector<size_t> shapes = {2, 2};
  model::base_lattice lat("square lattice", "simple2d", shapes,
                          "../config/lattices.xml", false);
  string ham_path =
      "../gtest/model_array/Heisenberg/2D/"
      "original/Jx_-0.3_Jy_0.5_Jz_0.8_hx_0.3_hz_0/H/";
  model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, zero_worm,
                             false, false, alpha);

  double T = 1;
  double beta = 1 / T;
  size_t sweeps;
  size_t therms;
  sweeps = 400000;
  therms = 0;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.emplace_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);
  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  EXPECT_FLOAT_EQ(out_res.ene.mean, -0.22672973371746891);
  EXPECT_NEAR(out_res.ene.mean, -0.22695394021770868,
              3 * out_res.ene.err);  // -0.18543629571195416 for L = [2,4] J =
                                     // [-0.3, 0.5, 0.8] hx = 0.3
}

TEST(HXX2D, MES_a_zw) {
  // alpha = 1 means local hamiltonian only contains single site.
  double alpha = 0.2;
  double shift = 0.1;
  std::vector<size_t> shapes = {2, 2};
  model::base_lattice lat("square lattice", "simple2d", shapes,
                          "../config/lattices.xml", false);
  string ham_path =
      "../gtest/model_array/Heisenberg/2D/"
      "original/mes/Jx_-0.3_Jy_0.5_Jz_0.8_hx_0.3_hz_0/M_10/H/";
  bool zero_worm = true;
  model::base_model<MC> spin(lat, {2}, ham_path, {1}, {0}, shift, zero_worm,
                             false, false, alpha);

  double T = 1;
  double beta = 1 / T;
  size_t sweeps;
  size_t therms;
  sweeps = 400000;
  therms = 0;

  size_t cutoff_l = std::numeric_limits<size_t>::max();
  model::MapWormObs mapwobs;

  vector<string> wobs_paths;
  wobs_paths.emplace_back("");
  std::vector<batch_res> res;
  model::observable obs(spin, "", false);

  mc_res out_res = run_worm(spin, T, sweeps, therms, res, obs, lat, mapwobs);

  EXPECT_FLOAT_EQ(out_res.ene.mean, -0.22710672923632713);
  EXPECT_NEAR(out_res.ene.mean, -0.22695394021770868,
              3 * out_res.ene.err);  // -0.18543629571195416 for L = [2,4] J =
                                     // [-0.3, 0.5, 0.8] hx = 0.3
}
