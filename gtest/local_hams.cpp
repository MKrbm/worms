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

#include "gtest/gtest.h"
#include "dataset.hpp"

using namespace std;

// TEST(HamsTest, SS_4x4_aggr) {
//   std::vector<size_t> shapes = {2, 2};
//   model::base_lattice lat("square lattice", "SS2", shapes, "../config/lattices.xml", true);
//   string ham_path = "../gtest/model_array/SS/H1";
//   model::base_model<bcl::st2013> spin(lat, {4, 4}, ham_path, {1,1}, {0,1}, 0.1, false, false, true);
//   cerr << lat.site_type << endl;
//   cerr << spin.num_types() << endl;
//   cerr << spin.nn_sites << endl;
//   cerr << spin.s_flip_max_weights << endl;
// }

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
  double alpha = 1 / 6.0;
model::base_lattice lat("triangular lattice", "anisotropic triangular", shapes, "../config/lattices.xml", false);
string ham_path = "../gtest/model_array/KH/smel/H1";
model::base_model<MC> spin(lat, {8}, ham_path, {1, 1, 1}, {0, 1, 2}, 0.1, false, false, false, alpha);


model::base_lattice lat2("triangular lattice", "kagome", shapes, "../config/lattices.xml", false);
string ham_path2 = "../gtest/model_array/KH/origina/none/H";
model::base_model<MC> spin2(lat2, {2}, ham_path2, {1}, {0}, 0.3, false, false, false);


TEST(HamsTest, Kagome4x4SplitTest2)
{
  auto lops = spin2.loperators;
  size_t sps = spin2.sps_sites(0);
  spin_state::StateFunc state_func(sps);
  spin_state::StateFunc bond_func(sps*sps);

  for (int i=0; i<lops.size(); i++){
    auto lop = lops[i];
    int size = lop.size;
    int N = size*size;
    for (int j=0; j<N; j++){
      auto index = state_func.num2state(j, 4);
      auto bond_index = bond_func.num2state(j, 2);
      if (bond_index[0] == bond_index[1]){
        double val = lop.ham_vector(j);
        val += lop.single_flip(0, index[1])[index[0]][index[0]] + lop.single_flip(1, index[0])[index[1]][index[1]];
        EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
      } else {
        if (index[0] == index[2]){
          double val = lop.single_flip(true, index[0])[index[1]][index[3]];
          EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
          EXPECT_FLOAT_EQ(val, 0);
        } else if (index[1] == index[3]){
          double val = lop.single_flip(false, index[1])[index[0]][index[2]];
          EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
          EXPECT_FLOAT_EQ(val, 0);
        } else {
          EXPECT_NEAR(lop.ham_vector(j) * lop.signs[j], lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
        }
      }
    }
  }
}


TEST(HamsTest, Kagome4x4SplitTest)
{
  auto lops = spin.loperators;
  size_t sps = 8;
  spin_state::StateFunc state_func(sps);
  spin_state::StateFunc bond_func(sps*sps);

  for (int i=0; i<lops.size(); i++){
    auto lop = lops[i];
    int size = lop.size;
    int N = size*size;
    for (int j=0; j<N; j++){
      auto index = state_func.num2state(j, 4);
      auto bond_index = bond_func.num2state(j, 2);
      if (bond_index[0] == bond_index[1]){
        double val = lop.ham_vector(j);
        val += lop.single_flip(0, index[1])[index[0]][index[0]] + lop.single_flip(1, index[0])[index[1]][index[1]];
        EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
      } else {
        if (index[0] == index[2]){
          double val = lop.single_flip(true, index[0])[index[1]][index[3]];
          EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
        } else if (index[1] == index[3]){
          double val = lop.single_flip(false, index[1])[index[0]][index[2]];
          EXPECT_NEAR(val, lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
        } else {
          EXPECT_NEAR(lop.ham_vector(j) * lop.signs[j], lop.ham_prime()[bond_index[0]][bond_index[1]], 1E-8);
        }
      }
    }
  }
}

TEST(HamsTest, Kagome4x4WormUpdate)
{
  double beta = 0.5;
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
  solver.initStates();
  double wdensity = 5;
  double wlength = 0;
  double wcount = 0;
  for (int i = 0; i < 1E3; i++)
  {
    solver.diagonalUpdate(wdensity);
    solver.wormUpdate(wcount, wlength);
  }
  cerr << "finish" << endl;
}


TEST(HamsTest, Kagome4x4DiagonalUpdate)
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
  solver.initStates();

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
    EXPECT_NEAR(bops[b].mean(), weights_bonds[b], 8 * bops[b].error());
  }

  for (int s = 0; s < sites.size(); s++){
    EXPECT_NEAR(sops[s].mean(), weights_sites[s], 8 * sops[s].error());
  }
  
  auto state0 = state;
  int x1 = 0;
  int x2 = 0;
  int sps = solver.sps;
  vector<double> markov_mat_elem(2 * sps, 0);
  state_t nn_state(0);
  int flip_site = 0;
  for (int site : sites[flip_site]){
    nn_state.push_back(state0[site]);
  }
  double sum = 0;
  int fl0 = static_cast<int>(uniform(rand_src) * (sps-1))+1;
  int dir0 = static_cast<int>(uniform(rand_src));
  for (int j=1; j < 2*sps; j++){
    OP_type op3 = OP_type(&sites[flip_site], &solver.pows_vec[3], state0[flip_site] + sps * state0[flip_site],nn_state, -1, 0);
    op3.update_state(dir0, (sps - fl0)%sps);
    int dir = j / sps;
    int fl = j % sps;
    int num = op3.update_state(dir, fl);
    markov_mat_elem[j] = std::abs(solver.get_single_flip_elem(op3));
    if (fl == 0) {
      markov_mat_elem[j] = 0;
    }
    sum += markov_mat_elem[j];
  }

  int fl = fl0;
  int dir = dir0;
  OP_type op2 = OP_type(&sites[flip_site], &solver.pows_vec[3], state0[flip_site] + sps * state0[flip_site],nn_state, -1, 0);
  int num_0 = op2.state();
  int num = num_0;
  int next_num = num;
  vector<double> markov_cnt(2 * sps, 0);
  int N = 5*1E5;
  for (int n=0; n < N; n++){
    dir = !dir;
    fl = (sps - fl)%sps;
    auto flip = solver.markov_next_flip(op2, !dir, fl, false);
    dir = flip.first;
    fl = flip.second;
    markov_cnt[fl + (sps) * dir] ++;
    if (num < 0 || num >= sps*sps){
      cerr << "num = " << num << endl;
    }
  }

  for (int j=1; j < 2*sps; j++){
    double p = markov_mat_elem[j] / sum;
    double var = p * (1-p) / N;
    EXPECT_NEAR(markov_cnt[j] / N, markov_mat_elem[j] / sum , 8 * sqrt(var));
  }
}

TEST(HamsTest, Kagome_4x4_aggr)
{
  double _shift = 0.1;
  double alpha = 1 / 6.0;
  model::base_model<MC> spin(lat, {8}, ham_path, {1, 1, 1}, {0, 1, 2}, _shift, false, false, false, alpha);
  spin_state::StateFunc state_func(8, 6);
  vector<double> shifts;

  for (int i = 0; i < spin.loperators.size(); i++)
  {
    vector<vector<double>> H = spin.loperators[i]._ham;
    double shift = 0;
    double max_elem = std::numeric_limits<double>::lowest();
    for (int i = 0; i < H.size(); i++)
    {
      shift = std::min(shift, H[i][i]);
      max_elem = std::max(max_elem, H[i][i]);
    }
    shift *= -1;
    shift += _shift;
    shifts.push_back(shift);
    EXPECT_FLOAT_EQ((max_elem + shift) * (1 - alpha), spin.loperators[i].max_diagonal_weight_);
  }
  // cerr << spin.nn_sites << endl;
  double shift = 0;
  double max_val = 0;
  for (int i = 0; i < 8; i++)
  {
    for (int j = 0; j < round(pow(8, 6)); j++)
    {
      state_t state = state_func.num2state(j, 6);
      double val = 0;
      val += spin.loperators[0]._ham[state[0] + i * 8][state[0] + i * 8];
      val += spin.loperators[1]._ham[state[2] + i * 8][state[2] + i * 8];
      val += spin.loperators[2]._ham[state[4] + i * 8][state[4] + i * 8];
      val += spin.loperators[0]._ham[state[1] * 8 + i][state[1] * 8 + i];
      val += spin.loperators[1]._ham[state[3] * 8 + i][state[3] * 8 + i];
      val += spin.loperators[2]._ham[state[5] * 8 + i][state[5] * 8 + i];
      val += 2 * (shifts[0] + shifts[1] + shifts[2]);
      max_val = std::max(max_val, val);
    }
  }

  EXPECT_FLOAT_EQ(max_val / 2 * alpha, spin.s_flip_max_weights[0]);

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
  double max_diag = 0;
  for (int i = 0; i < spin.loperators.size(); i++)
  {
    max_diag = std::max(max_diag, spin.loperators[i].max_diagonal_weight_);
  }

  for (int i = 0; i < num_type(spin.site_type); i++)
  {
    max_diag = std::max(max_diag, spin.s_flip_max_weights[i]);
  }
  EXPECT_FLOAT_EQ(solver.rho / (spin.Nb + spin.L), max_diag);
}

TEST(HamsTest, Kagome_2x2_array)
{
  vector<vector<double>> H = spin.loperators[0]._ham;
  //* diagonal element at site = (0, 1) spins are = (1, 0)

  spin_state::StateFunc state_func(8, 2);
  int x = state_func.state2num({1, 0}, 2);

  //! Note that the indexing rule of numpy is different from this.
  //! In numpy, {1,0} will be 8 but here it is 1.
  EXPECT_EQ(x, 1);
  EXPECT_FLOAT_EQ(H[x][x], 0.16666234482156317);

  // d* test single flip operator
  //* end spin is fixed to 3
  auto single_flip = spin.loperators[0].single_flip(0, 3);

  double shift = 0;
  for (int i = 0; i < H.size(); i++)
  {
    shift = std::min(shift, H[i][i]);
  }
  shift *= -1;
  shift += 0.1;
  EXPECT_FLOAT_EQ(shift, spin.loperators[0].ene_shift);
  EXPECT_FLOAT_EQ(single_flip[1][3], -0.11785105507860531);

  // d* diagonal element is little bit tricky
  double elem2 = -0.16666596146430332;
  elem2 += shift;
  elem2 = abs(elem2);
  EXPECT_FLOAT_EQ(spin.loperators[0].ham_prime()[2 + 3 * 8][2 + 3 * 8], elem2);
  EXPECT_FLOAT_EQ(single_flip[2][2], elem2 / 2 * alpha);

  //* fix end site
  auto single_flip2 = spin.loperators[0].single_flip(1, 2);
  EXPECT_FLOAT_EQ(single_flip2[1][3], -0.11785629850799827);
  elem2 = -0.33333079565882834;
  elem2 += shift;
  elem2 = abs(elem2);
  EXPECT_FLOAT_EQ(spin.loperators[0].ham_prime()[2 + 4 * 8][2 + 4 * 8], elem2);
  EXPECT_FLOAT_EQ(single_flip2[4][4], elem2 / 2 * alpha);
}
