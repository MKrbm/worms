#ifndef __loop__
#define __loop__


#pragma once
#include <string.h>
#include <iostream>
#include <uftree.hpp>
#include <vector>
#include <random>
#include <fstream>
#include <ostream>
#include <strstream>
#include <sstream>
#include <algorithm>

#include <bcl.hpp>

#ifdef __APPLE__
#  include <mach-o/dyld.h>
#endif
#if defined(_WIN32)
#  include <windows.h>
#else
  #include <unistd.h>
#endif


#include <filesystem>
#include <unistd.h>

#include <ctime>
#include <math.h> 

#include "state.hpp"
#include "model.hpp"
#include "BC.hpp"
#define SEED 2021
/* inherit UnionFindTree and add find_and_flip function*/

// template <typename MODEL>

inline int positive_modulo(int i, int n) {
    return (i % n + n) % n;
}
using MODEL = model::heisenberg1D;
using OPS = std::vector<spin_state::OpStatePtr>;
using BSTATE = spin_state::BottomState;
using WORMS = spin_state::Worms;
using DOTS = std::vector<spin_state::Dot>;




class worm{
  public:
  MODEL model;
  double beta;
  int L;
  int W; //number of worms
  OPS ops_main; //contains operators.
  OPS ops_sub; // for sub.


  // OPS& ops_main = ops_tmp1;  //contains operators.
  // OPS& ops_sub = ops_tmp2; // for sub.
  spin_state::BStatePtr pstate;
  spin_state::WormsPtr pworms;

  spin_state::Worms& worms = *pworms;
  spin_state::BottomState& state = *pstate;
  std::vector<int> worms_label;


  // BSTATE& state;
  // WORMS& worms;

  DOTS spacetime_dots; //contain dots in space-time.

  std::vector<double> worms_tau;
  std::vector< std::vector<int> > bonds;

  //declaration for random number generator
  // typedef model::local_operator::engine_type engine_type;
  typedef std::mt19937 engine_type;
  #ifdef RANDOM_SEED
  engine_type rand_src = engine_type(static_cast <unsigned> (time(0)));
  #else
  engine_type rand_src = engine_type(SEED);
  #endif


  // random distribution from 0 to 1
  std::uniform_real_distribution<> uni_dist; 
  typedef std::uniform_real_distribution<> uniform_t;
  typedef std::exponential_distribution<> expdist_t;
  uniform_t uniform;
  // reference of member variables from model class

  static const int N_op = MODEL::Nop;

  std::array<model::local_operator, N_op>& loperators; //holds multiple local operators
  std::array<int, N_op>& leg_sizes; //leg size of local operators;
  std::array<double, N_op>& operator_cum_weights;



  worm(double beta, MODEL model_)
  :model(model_), L(model.L), beta(beta),
  // dist(0, model.Nb-1), worm_dist(0.0, beta),
  bonds(model.bonds), pworms(spin_state::WormsPtr(new WORMS(20))),
  pstate(spin_state::BStatePtr(new BSTATE(L))), 
  loperators(model.loperators), leg_sizes(model.leg_size),
  operator_cum_weights(model.operator_cum_weights)
  {
    cout << "beta : " << beta << endl;
  }


  //* functions for initializing
  /* initialize worms */
  // void init_worms_rand(){
  //   worms_label.resize(0);
  //   for (int i=0; i<W; i++){
  //     // pworms->worm_site[i] = dist(rand_src);
  //     // pworms->tau_list[i] =  worm_dist(rand_src);

  //     worms.worm_site[i] = dist(rand_src);
  //     worms.tau_list[i] =  worm_dist(rand_src);
  //   }
  //   std::sort(worms.tau_list.begin(), worms.tau_list.end(), std::less<double>());
  // }

  void init_states(){ //* initialized to all up
  for (auto& x : state){
    x = 0;
    }
  }

  void init_dots(bool add_state = true){
    spacetime_dots.resize(0);
    if (add_state){
      for(int i=0; i<L; i++){
        set_dots(i, 0, i);
      }
    }
  }

  //swapt main and sub
  void swap_oplist(){
    // auto tmp_op = ops_sub;
    // ops_sub = ops_main;
    // ops_main = tmp_op;
    ops_main.swap(ops_sub);
  }


  //main functions

  void diagonal_update(double wdensity){
    
    int n_worm = 0;
    // init_ops_main()
    ops_main.resize(0);
    //init spacetime_dots
    init_dots();
    //init worms
    worms.worm_site.resize(0);
    worms.tau_list.resize(0);
    worms_label.resize(0);

    double tmp = model.rho * beta + wdensity;
    expdist_t expdist(model.rho * beta + wdensity); //initialize exponential distribution
    double pstart = wdensity / (beta * model.rho + wdensity); //probability of choosing worms

    std::size_t N_op = model.Nop;

    std::size_t s0, s1;
    int r_bond; // randomly choosen bond
    std::vector<int> cstate = state;
    std::vector<int> local_state;
    double max_, target;
    std::size_t lop_label, s_num;
    lop_label = 0; //typically, lop_label is fixed to 0
    int leg_size = leg_sizes[lop_label]; //size of choosen operator
    auto& lop = loperators[lop_label];

    ops_sub.emplace_back(new spin_state::OpState((double)1)); //*sentinels
      
    double tau = expdist(rand_src);
    for (OPS::iterator opi = ops_sub.begin(); opi != ops_sub.end();){
      auto op_sub = *opi;
      if (tau < op_sub->tau){ //* if new point is behind the next operator is opsub.
        double r = uniform(rand_src);

        if (r < pstart){
          int s = static_cast<int>(model.L * uniform(rand_src));
          worms.worm_site.push_back(s);
          worms.tau_list.push_back(tau);
          worms[n_worm] = cstate[s];
          set_dots(s, 2 , worms.size());
          n_worm++;
        }else{
          int b = static_cast<int>(bonds.size() * uniform(rand_src));
          const auto& bond = bonds[b];
          int u = spin_state::state2num(cstate, bond);
          r = uni_dist(rand_src);
          if (r < lop.accept[u]){
            local_state = spin_state::num2state(u + (u<<leg_size), 2*leg_size);
            append_ops(ops_main, local_state, &lop, bond, tau);
            for (int i=0; i<leg_size; i++){
              set_dots(bond[i], 1 , i);
            }
          }
        }
        tau += expdist(rand_src);
      }else{ //*if tau went over the operator time.
        if (op_sub->is_off_diagonal()) {
          update_state(op_sub, cstate);
          ops_main.push_back(op_sub);
          for (int i=0; i<op_sub->L; i++){
            set_dots(op_sub->bond[i], 1 , i);
          }
        }
        ++opi;
      }
    } //end of while loop
  }
  /*
  * check off-diagonal operator in ops_sub and flip accordingly.
  * note that diagonal operators will disappear during this step.
  */
  void checkODNFlip(double& optau, double tau_prime, std::size_t& op_label,
                     std::vector<int>& cstate ){
    while(optau<tau_prime && op_label < ops_sub.size()){
      auto op_ptr = ops_sub[op_label];
      // spin_state::BaseStatePtr op_ptr_ = ops_sub[op_label];
      spin_state::BaseStatePtr stateptr;
      stateptr = ops_sub[op_label];
      // cout << "L : " << stateptr->plop->L << endl;
      if (op_ptr->is_off_diagonal()){
        // update_state_OD(op_ptr, cstate);
        update_state(op_ptr, cstate);
        ops_main.push_back(op_ptr);
        for (int i=0; i<op_ptr->L; i++){
          set_dots(op_ptr->bond[i], 1 , i);
        }
      }
      op_label++;
      if (op_label < ops_sub.size()) optau = ops_sub[op_label]->tau;
    }
    return;
  }

  //*append to ops
  void append_ops(OPS& ops, std::vector<int> const& ls,
    model::local_operator* plop, std::vector<int> const& bond, double tau){
    ops_main.emplace_back( new spin_state::OpState(ls, plop, bond, tau));
  }
  //*overload for r value
  void append_ops(OPS& ops, std::vector<int>&& ls,
    model::local_operator* plop, std::vector<int>&& bond, double tau){
    ops_main.emplace_back( new spin_state::OpState(ls, plop, bond, tau));
  }
 
  /*
  *perform one step from given worm.
  If dot is operator then, worm move to exit. otherwise just assigin spin to dot.
  params
  ------
  int next_dot : next dot.
  int dir : direction worm is moving toward. 1 : move up, 0 : move down.
  int spin : current spin state of worm.
  int site : site worm is at.

  params(member variables)
  ------
  */
  void worm_process_op(int& next_dot, int& dir, int& spin, int& site){

    int clabel = next_dot;
    spin_state::Dot& dot = spacetime_dots[clabel];
    int dtype = dot.dot_type;

    ASSERT(site == dot.site, "site is not consistent");
    if (dtype!=1){ //n* if dot is state or worm.
      *dot.sptr = spin;
      // next_dot = dot.move_next(dir);
      return;
    }

    if (dtype==1){
      int dir_in = !dir; //n* direction the worm comes in from the view of operator.
      std::size_t cindex = dot.typeptr->GetIndex(dot.sptr, dir_in);
      auto& opstate = *dot.typeptr;
      opstate[cindex] ^= 1;
      int num = spin_state::state2num(opstate, opstate.get_size());
      double r = uni_dist(rand_src);
      int nindex = opstate.plop->markov[num](cindex, rand_src);
      opstate[nindex] ^= 1;

      //n* assigin for next step
      dir = nindex/(opstate.L);
      site = opstate.bond[nindex%opstate.L];
      spin = opstate[nindex];

      next_dot = opstate.GetLabel(cindex, nindex, clabel);

      // opstate.plop->tran
    }
  }

  /*
  *update worm for W times.
  */
  void worm_update(){
    int w_index =0;
    int dots_size = spacetime_dots.size();
    for (auto w_label : worms_label){
      int d_label = w_label;
      auto* dot_ptr = &spacetime_dots[d_label];
      int site = dot_ptr->site;
      int dir = 2 * uniform(rand_src);//n initial direction is 1.
      *dot_ptr->sptr = 1^*dot_ptr->sptr; //n flip the worm. and it propagate through spacetime.
      int spin = *dot_ptr->sptr;
      do{
        check_operators_while_update(w_label, dir ? d_label : dot_ptr->prev);
        d_label = dot_ptr->move_next(dir);
        worm_process_op(d_label, dir, spin, site);
        dot_ptr = &spacetime_dots[d_label];
      }while(d_label != w_label);
      w_index++;  
    }
  }

  /*
  *this function will be called after assigining op_main
  */
  void set_dots(int site, int dot_type, int index){

    int* sptr;
    spin_state::BaseStatePtr stateptr;

    // ASSERT(label == spacetime_dots.size()+1)
    int label = spacetime_dots.size();

    int prev, end;
    if (dot_type == 0) {
      sptr = state.data() + index;
      stateptr = pstate;
      ASSERT(label == site, "label must be equal to site");
      spacetime_dots.emplace_back(
        site, site, site, sptr,
        stateptr, dot_type
      );
    }else if(dot_type == 1){
      int n = ops_main.size();
      sptr = ops_main[n-1]->data() + index;
      stateptr = ops_main[n-1];
      spacetime_dots.emplace_back(
        site, spacetime_dots[site].prev, site, sptr,
        stateptr, dot_type
      );
      spacetime_dots[spacetime_dots[site].prev].set_next(label);
      spacetime_dots[site].set_prev(label);
    }else if(dot_type == 2){
      sptr = worms.data() + index;
      stateptr = pworms;
      worms_label.push_back(label);
      spacetime_dots.emplace_back(
        site, spacetime_dots[site].prev, site, sptr,
        stateptr, dot_type
      );
      spacetime_dots[spacetime_dots[site].prev].set_next(label);
      spacetime_dots[site].set_prev(label);
    }

  }

  /*
  *update given state by given operator ptr;
  */
  static void update_state(spin_state::BaseStatePtr op_ptr, std::vector<int>& state){
    std::vector<int> local_state = *op_ptr;
    std::vector<int> state_(op_ptr->plop->L);
    int i=0;
    for (auto x : op_ptr->bond){
      state_[i] = state[x];
      i++;
    }
    ASSERT(is_same_state(local_state, state_), "the operator can not be applied to the state");
    if (op_ptr->is_off_diagonal()) update_state_OD(op_ptr, state);
  }

  /*
  *update given state by given offdiagonal operator ptr;
  */
  static void update_state_OD(spin_state::BaseStatePtr op_ptr, std::vector<int>& state){
    int index = 0;
    for (auto x : op_ptr->bond){
      state[x] = (*op_ptr)[op_ptr->L + index];
      index++;
    }
  }
  /*
  * check the operator and state is consistent during the worm_updateg
  params
  ------
  worm_label : label of dot (worm) we are aiming at.
  p_label : label of dot before reaching at the current position of worm.

  */
  void check_operators_while_update(int worm_label, int p_label){
    
    #ifndef NDEBUG
    auto state_ = state;
    spin_state::BaseStatePtr ptr = nullptr;

    int label = 0;
    for (const auto& dot:spacetime_dots){


      if (dot.dot_type == 1 && ptr != dot.typeptr){ //if dot_type is operator
        ptr = dot.typeptr;  
        update_state(ptr, state_);
      }
      if (dot.dot_type != 1){
        int spin = worm_label == label ? 1^(*dot.sptr) : (*dot.sptr);
        ASSERT(state_[dot.site] == spin, "spin is not consistent");
        state_[dot.site] = *dot.sptr;
      }

      if (p_label == label){
        state_[dot.site] = 1^state_[dot.site];
      }

      label++;
    }
    ASSERT(is_same_state(state_, state), "operators are not consistent while update worms");
    #endif 
    return;
  }


  //* functions for testing
  static void check_operators(spin_state::BottomState state, OPS ops){
    #ifndef NDEBUG
    const auto state_ = state;
    for (auto op : ops){
      update_state(op, state);
    }
    ASSERT(is_same_state(state_, state), "operators are not consistent");
    #endif
    return;
  }

  static bool is_same_state(int n, int m){
    return n==m;
  }

  static bool is_same_state(int n, std::vector<int> state){
    int m = spin_state::state2num(state, state.size());
    return n==m;
  }

  static bool is_same_state( std::vector<int> state_, std::vector<int> state){
    int m = spin_state::state2num(state, state.size());
    int n= spin_state::state2num(state_, state.size());
    return n==m;
  }


};

#endif 