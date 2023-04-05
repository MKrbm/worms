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
uniform_t uniform;
int seed = static_cast<unsigned>(time(0));
auto rand_src = engine_type(seed);

std::vector<size_t> shapes = {4, 4};
model::base_lattice lat("triangular lattice", "anisotropic triangular", shapes, "../config/lattices.xml", false);
string ham_path = "../gtest/model_array/KH/smel/H1";
model::base_model<MC> spin(lat, {8}, ham_path, {1, 1, 1}, {0, 1, 2}, 0.1, false, false, true);

TEST(HamsTest, Kagome_4x4_aggr)
{
  spin_state::StateFunc state_func(8, 6);
  vector<double> shifts;
  double alpha = 1.0 / 6;

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
    shift += 0.1;
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
      VUS state = state_func.num2state(j, 6);
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

  double wdensity = 5;
  expdist_t expdist(solver.rho * beta + wdensity);
  double pstart = wdensity / (beta * solver.rho + wdensity);
  BC::observable p;
  std::vector<pair<BC::observable, double>> bops(spin.Nb);
  std::vector<pair<BC::observable, double>> sops(spin.L);


  vector<unsigned short> state(16, 0);
  for (auto& s : state)
  {
    s = static_cast<unsigned short>(uniform(rand_src) * 8);
  }
  size_t sweeps = 1E5;
  for (int i = 0; i < sweeps; i++)
  {
    double tau = expdist(rand_src);
    double _p = 0;
    double _bo = 0;
    double _so = 0;
    while (tau < 1)
    {
      double r = uniform(rand_src);

      if (r < pstart)
      {
        // _p++;
        p << 1;
        //* append worm
      }
      else
      {
        size_t b = static_cast<size_t>((spin.Nb + spin.L) * uniform(rand_src));
        r = uniform(rand_src);
        if (b < spin.Nb)
        {
          //* append bond ops
          double bop_label = solver.bond_type[b];
          auto const &accept = solver.accepts[bop_label];
          auto const &bond = solver.bonds[b];
          size_t u = solver.state_funcs[bop_label].state2num(state, bond);
          if (r < accept[u])
          {
            //* append accept
            // _bo += 1.0 / spin.Nb;
            bops[b].first << 1.0;
            bops[b].second = accept[u] * max_diag;
          }
        }
        else
        {
          //* append single-flip
          int site = b - spin.Nb;
          double sop_label = spin.site_type[site];
          double mat_elem = 0;
          for (auto target : spin.nn_sites[site]){
            mat_elem += spin.loperators[target.bt]
              .single_flip(target.start, state[target.target], state[site], state[site]);
          }

          mat_elem = std::abs(mat_elem) / max_diag;
          if (mat_elem > 1) throw std::runtime_error("mat_elem > 1");
          if (r < mat_elem)
          {
            //* append accept
            sops[site].first << 1.0;
            sops[site].second = mat_elem * max_diag;
          }
        }
      }
      tau += expdist(rand_src);
    }

  }

  EXPECT_NEAR(p.count() / (double) sweeps, wdensity, 1E-2 * wdensity);
  for (auto &b : bops)
  {
    EXPECT_NEAR(b.first.count() / (double) sweeps, b.second, 1E-2);
  }

  for (auto &s : sops)
  {
    EXPECT_NEAR(s.first.count() / (double) sweeps, s.second, 1E-2);
  }

}


TEST(HamsTest, Kagome_4x4_diagonal_update)
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
  solver.diagonalUpdate(wdensity);
}


TEST(HamsTest, Kagome_2x2_array)
{
  std::vector<size_t> shapes = {2, 2};
  model::base_lattice lat("triangular lattice", "anisotropic triangular", shapes, "../config/lattices.xml", false);
  string ham_path = "../gtest/model_array/KH/smel/H1";
  model::base_model<bcl::st2013> spin(lat, {8}, ham_path, {1, 1, 1}, {0, 1, 2}, 0.1, false, false, true);
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
  double alpha = 1 / 6.0;
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