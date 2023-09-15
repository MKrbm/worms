#pragma once
#include <string.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <ostream>
#include <strstream>
#include <sstream>
#include <algorithm>
#include <utility>
#include <unordered_set>
#include <bcl.hpp>
#include <stdlib.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <filesystem>
#include <unistd.h>

#include <ctime>
#include <math.h>

#include "state.hpp"
#include "operator.hpp"
#include "automodel.hpp"
#include "funcs.hpp"
#include "autoobservable.hpp"
#define SEED 34
/* inherit UnionFindTree and add find_and_flip function*/

// template <typename MODEL>

inline int positive_modulo(int i, int n)
{
  return (i % n + n) % n;
}

using spin_state::Dotv2;

// using MODEL = model::heisenberg1D;
using state_t = spin_state::state_t;
using SPIN = spin_state::US;
using BOND = model::VS;
using WORMS = spin_state::WORM_ARR;
using DOTS = std::vector<Dotv2>;
using size_t = std::size_t;

template <class MCT>
class Worm
{

public:
  using MODEL = model::base_model<MCT>;
  using LOPt = model::local_operator<MCT>;

private:
  model::MapWormObs _mp_worm_obs;
  alps::alea::batch_acc<double> _phys_cnt;

  // n*  number of physically meaningful configurations;
  double phys_cnt;
  // n*  sum of observables encountered while worm update. (observable must be non-diagonal operator)
  std::vector<double> obs_sum;
  
  //n* maximum diagonal value of local operator.

public:
  typedef spin_state::Operator OP_type;
  typedef std::vector<OP_type> OPS;
  typedef spin_state::StateFunc state_func;
  typedef std::mt19937 engine_type;
  typedef std::uniform_real_distribution<> uniform_t;
  typedef std::exponential_distribution<> expdist_t;
  typedef bcl::markov<engine_type> markov_t;
  double max_diagonal_weight = -1;

  uniform_t uniform;

  engine_type rand_src;
  engine_type test_src;

  MODEL spin_model;
  OPS ops_main; // n* contains operators.
  OPS ops_sub;  // n*  for sub.
  state_t state;
  state_t cstate;
  DOTS spacetime_dots; // n*  contain dots in space-time.
  WORMS worms_list;

  VVS pows_vec;
  size_t sps;
  std::vector<BOND> bonds;
  std::vector<BOND> nn_sites;
  std::vector<size_t> bond_type;
  std::vector<state_func> state_funcs;
  std::unordered_set<size_t> can_warp_ops;
  std::vector<size_t> pres = std::vector<size_t>(0);
  std::vector<size_t> psop = std::vector<size_t>(0);
  std::vector<state_t> worm_states;
  std::vector<double> worm_taus;


  // n* reference of member variables from model class
  std::vector<model::local_operator<typename MODEL::MCT>> &loperators;
  std::vector<std::vector<double>> accepts; // n* normalized diagonal elements;

  // end of define observables

  int sign = 1;
  int cnt = 0;
  const int L; // n* number of sites
  const int N_op;
  size_t d_cnt = 0;
  size_t bocnt = 0;
  const size_t cutoff_length; // n* cut_off length
  size_t u_cnt = 0;
  double rho;
  const double beta;
  bool zw;


  std::unordered_map<std::string, WormObs> &get_worm_obs() { return _mp_worm_obs(); }
  alps::alea::batch_acc<double> &get_phys_cnt() { return _phys_cnt; }

  Worm(double beta, MODEL model_, size_t cl = SIZE_MAX, int rank = 0, int seed = SEED)
    : Worm(beta, model_, model::WormObs(model_.sps_sites(0)), cl, rank, seed) {}

  Worm(double beta, MODEL model_, model::MapWormObs mp_worm_obs_, size_t cl = SIZE_MAX, int rank = 0, int seed = SEED);

  inline void initStates()
  { //* initialized to all up
    for (int i = 0; i < state.size(); i++)
    {
      double r = uniform(rand_src);
      state[i] = static_cast<SPIN>(sps * r);
    }
  }

  inline void initDots(bool add_state = true)
  {
    spacetime_dots.resize(0);
    if (add_state)
    {
      for (int i = 0; i < L; i++) set_dots(i, -1, i);
    }
  }

  // swapt main and sub
  inline void swapOps(){ std::swap(ops_main, ops_sub);}
  //*append to worms
  inline void appendWorms(WORMS &wm, size_t site, size_t dot_label, double tau) 
    { wm.push_back(std::make_tuple(site, dot_label, tau));}
  
  inline bool is_same_state(int n, int m) { return n == m;}
  inline bool is_same_state(int n, state_t state, size_t lopt) {return n == state_funcs[lopt].state2num(state, state.size());}
  inline bool is_same_state(state_t state_, state_t state, size_t lopt)
  {
    int m = state_funcs[lopt].state2num(state, state.size());
    int n = state_funcs[lopt].state2num(state_, state.size());
    return n == m;
  }
  void diagonalUpdate(double wdensity);

  /*
  update Worm W times.
  variables
  ---------
  dir : direction of worm head. 1 : upward, -1 : downward
  */
  void wormUpdate(double &wcount, double &wlength);

  /*
  This function will be called ever time the head of the worm cross the same propagation level.
  calculate $\langle x_{\tau} | \hat{o}_i \hat{o}_j |x^\prime_{\tau} \rangle$ and update the state of the worm. $x is lower states and x^\prime is upper states$

  note that this function is only called when the worm position of head and tail is apart.
  params
  ------
  tau : imaginary time of the worm head and tail.
  h_site : site of the worm head.
  t_site : site of the worm tail.
  h_x : lower state of the worm head.
  h_x_prime : upper state of the worm head.
  t_w : lower state of the worm tail.
  t_x_prime : upper state of the worm tail.
  */
  void calcHorizontalGreen(double tau, size_t h_site, size_t t_site,
                           size_t h_x, size_t h_x_prime, size_t t_x, size_t t_x_prime,
                           const state_t &_cstate);

  /*
  This function will be called ever time worm head warps.
  */
  void calcWarpGreen(double tau, size_t t_site, size_t t_x, size_t t_x_prime, const state_t &_cstate);

  // //*append to ops
  void appendOps(
      OPS &ops,
      DOTS &sp,
      std::unordered_set<size_t> &warp_sp,
      const BOND *const bp,
      const BOND *const pp,
      int state,
      int op_type,
      double tau);

  // //*append to ops
  void appendSingleOps(
      OPS &ops,
      DOTS &sp,
      std::unordered_set<size_t> &warp_sp,
      int s_site,
      const BOND *const bp,
      const BOND *const pp,
      int state,
      const state_t& nn_state,
      int op_type,
      double tau);


  //* get dot state
  /*
  params
  ------
  ndot_label : label of the dot worm is directing to.
  dir : direction of the worm moves to dot. 1 : upwards, 0 : downwards. So if dir = 1, it means worm comes from below the dot.
  */
  size_t getDotState(size_t ndot_label, size_t dir);


  /*
  *perform one step from given Worm.
  If dot is operator then, Worm move to exit of the operator. otherwise just assigin spin to dot.
  params
  ------
  int next_dot : next dot.
  int dir : direction Worm is moving toward. 1 : move up, 0 : move down.
  int spin : current spin state of Worm.
  int site : site Worm is at.

  params(member variables)
  ------
  */
  int wormOpUpdate(int &next_dot, int &dir,
                   int &site, double &wlength, int &fl, double &tau,
                   const int wt_dot, const int wt_site, const double wt_tau,
                   int &w_x, int &wt_x, const int t_fl, const int t_dir,
                   const int w_index);
  /*
   *this function will be called after assigining op_main
   */
  void set_dots(size_t site, size_t dot_type, size_t index);
  void getSpinsDot(size_t next_dot, Dotv2 *dotp, int dir, size_t &h_x, size_t &h_x_prime);


  /*
   *update given state by given operator ptr;
   */
  void update_state(typename OPS::iterator opi, state_t &state);

  /*
   *update given state by given offdiagonal operator ptr;
   */
  void update_state_OD(typename OPS::iterator opi, state_t &state);

  /*
  * check the operator and state is consistent during the worm_updateg
  params
  ------
  worm_label : label of dot (Worm) we are aiming at.
  p_label : label of dot before reaching at the current position of Worm.

  */
  void checkOpsInUpdate(int worm_label, int p_label, int t_dir, int t_fl, int fl, int dir);
  bool detectWormCross(double tau, double tau_prime, double wt_tau, int dir);
  void reset_ops();
  double get_single_flip_elem(const OP_type& op);
  double get_single_flip_elem(int site, int x, int x_prime, state_t _state);
  double get_single_flip_elem(int site, int x, int x_prime, state_t _state, state_t& nn_state);


  /*
  params
  ------
  x : spin state of the site. x // sps is upper spin and x % sps is lower spin.
  x_flipped : spin state after flipped. x // sps is upper spin and x % sps is lower spin.
  fl : if fl = 0, then worm doesn't change state.

  comments
  --------
  x_flipped wax originally x but it is changed to x_flipped after worm comes to the operator and flip either upper or lower spin. 

  Original spin would be if direction is 1 (worm moves upwards) {x % sps + fl % sps,   x // sps}.
  */
  pair<int, int> markov_next_flip(OP_type& op, int dir, int fl, bool zero_fl = false);
  pair<int, int> markov_diagonal_nn(OP_type& op, int dir, int fl, int nn_index);
  static void printStateAtTime(const state_t &state, double time);
};

extern template class Worm<bcl::heatbath>;
extern template class Worm<bcl::st2010>;
extern template class Worm<bcl::st2013>;
