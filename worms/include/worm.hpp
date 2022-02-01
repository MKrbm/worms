#ifndef __loop__
#define __loop__


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
#define SEED 1643698133
/* inherit UnionFindTree and add find_and_flip function*/

// template <typename MODEL>

inline int positive_modulo(int i, int n) {
    return (i % n + n) % n;
}


using spin_state::Operatorv2;
using spin_state::Dotv2;

// using MODEL = model::heisenberg1D;
using STATE = model::STATE;
using SPIN = model::SPIN;
using BOND = model::BOND;
using WORMS = spin_state::WORM_ARR;
using DOTS = std::vector<Dotv2>;
using size_t = std::size_t;


template <typename MODEL>
class worm{
  public:
  typedef typename MODEL::base_spin_model base_spin_model;
  static const size_t nls = base_spin_model::nls;
  static const size_t max_L = base_spin_model::max_L;
  static const size_t sps = 1<<nls;
  typedef Operatorv2<nls, max_L> OP_type;
  typedef std::vector<OP_type> OPS;
  typedef spin_state::state_func<nls> state_func;

  MODEL model;
  OPS ops_main; //contains operators.
  OPS ops_sub; // for sub.
  STATE state;
  STATE cstate;
  DOTS spacetime_dots; //contain dots in space-time.
  WORMS worms_list;

  std::vector< BOND > bonds;
  std::vector<size_t> bond_type;
  
  std::vector<size_t> pres = std::vector<size_t>(0);
  std::vector<size_t> psop = std::vector<size_t>(0);
  // std::vector<size_t> st_cnt = std::vector<size_t>(0);
  double beta;
  size_t d_cnt=0;
  int L;

  //declaration for random number generator
  // typedef model::local_operator::engine_type engine_type;
  typedef std::mt19937 engine_type;
  #ifndef NDEBUG
  engine_type test_src = engine_type(SEED);
  #endif
  #ifdef RANDOM_SEED
  unsigned rseed = static_cast <unsigned> (time(0));
  engine_type rand_src = engine_type(SEED);
  #else
  engine_type rand_src = engine_type(SEED);
  #endif


  // random distribution from 0 to 1
  typedef std::uniform_real_distribution<> uniform_t;
  typedef std::exponential_distribution<> expdist_t;
  uniform_t uniform;
  // reference of member variables from model class

  static const int N_op = MODEL::Nop;
  // static const int N_op = 1;

  std::array<model::local_operator, N_op>& loperators; //holds multiple local operators
  std::array<int, N_op>& leg_sizes; //leg size of local operators;
  double rho;
  model::local_operator lop;
  std::vector<std::vector<double>> accepts; //normalized diagonal elements;

  typedef bcl::markov<engine_type> markov_t;
  std::vector<markov_t> markov;

  worm(double beta, MODEL model_)
  :model(model_), L(model.L), beta(beta), rho(-1),
  // dist(0, model.Nb-1), worm_dist(0.0, beta),
  bonds(model.bonds),bond_type(model.bond_type) ,state(model.L),cstate(model.L),
  loperators(model.loperators), leg_sizes(model.leg_size),
  lop(loperators[0]),markov(lop.markov)
  {
    cout << "beta        : " << beta << endl;
    #ifdef RANDOM_SEED
    cout << "seed number : " << rseed << endl;
    #endif
    double max_diagonal_weight = loperators[0].max_diagonal_weight_;
    for (auto const& lop : loperators){
      max_diagonal_weight = std::max(max_diagonal_weight, lop.max_diagonal_weight_);
    }
    for (int i=0; i<loperators.size(); i++){
      auto const& lop = loperators[i];
      auto accept = std::vector<double>(lop.size, 0);

      auto const& ham = lop.ham_;
      for (int j=0; j<lop.size; j++) {
        accept[j] = ham[j][j]/max_diagonal_weight;
      }
      accepts.push_back(accept);
      rho = max_diagonal_weight * model.lattice.num_bonds();
  }
  }


  void init_states(){ //* initialized to all up
  for (auto& x : state){
    #ifdef RANDOM_SEED
    x = static_cast<SPIN>(sps * uniform(rand_src));
    
    #else
    x = 1;
    #endif
    }
  }

  void init_dots(bool add_state = true){
    spacetime_dots.resize(0);
    if (add_state){
      for(int i=0; i<L; i++){
        set_dots(i, -1, i);
      }
    }
  }

  //swapt main and sub
  void swap_oplist(){
    std::swap(ops_main, ops_sub);
  }

  //main functions

  void diagonal_update(double wdensity){
    
    swap_oplist();
    // wdensity = 3;
    
    expdist_t expdist(rho * beta + wdensity); //initialize exponential distribution
    double pstart = wdensity / (beta * rho + wdensity); //probability of choosing worms
    std::copy(state.begin(), state.end(), cstate.begin());
    size_t lop_label;
    lop_label = 0; //typically, lop_label is fixed to 0
    // int leg_size = leg_sizes[lop_label]; //size of choosen operator
    // auto const& lop = loperators[lop_label];

    ops_main.resize(0); //* init_ops_main()
    
    init_dots(); //*init spacetime_dots

    //*init worms
    worms_list.resize(0);
    ops_sub.push_back(OP_type::sentinel(1)); //*sentinels
    double tau = expdist(rand_src);
    for (typename OPS::iterator opi = ops_sub.begin(); opi != ops_sub.end();){
      // auto op_sub = *opi;
      if (tau < opi->tau()){ //* if new point is behind the next operator is opsub.
        double r = uniform(rand_src);

        if (r < pstart){
          size_t s = static_cast<int>(L * uniform(rand_src));
          append_worms(worms_list, s, spacetime_dots.size(), tau);
          set_dots(s, -2 , 0); //*index is always 0 
        }else{
          size_t b = static_cast<size_t>(bonds.size() * uniform(rand_src));
          lop_label = bond_type[b];
          auto const& accept = accepts[lop_label];
          auto const& bond = bonds[b];

          // size_t u = spin_state_t::c2u(cstate[bond[0]], cstate[bond[1]]);
          size_t u = state_func::state2num(cstate, bond);
          // size_t u = spin_state::state2num(cstate, bond);

          r = uniform(rand_src);
          if (r < accept[u]){
            append_ops(ops_main, spacetime_dots, &bond, (u<<bond.size() * nls) | u, lop_label, tau);
          }
        }
        tau += expdist(rand_src);
      }else{ //*if tau went over the operator time.
        if (opi->is_off_diagonal()) {
          // auto const& bond = *opi->bond_ptr();
          update_state(opi, cstate);
          append_ops(ops_main, spacetime_dots, opi->bond_ptr(), opi->state(), opi->op_type(),opi->tau());
          printStateAtTime(cstate, tau);
        }
        ++opi;
      }
    } //end of while loop

    #ifndef NDEBUG
    std::cout << "bond \n\n" ;
    for (typename OPS::iterator opi = ops_main.begin(); opi != ops_main.end();++opi){
      printf("[%lu, %lu]\n", opi->bond(0), opi->bond(1));
    }
    #endif 
  }

  // //*append to ops
  static void append_ops(OPS& ops, DOTS& sp, const BOND * const bp, int state, int op_type, double tau){

    int s = bp->size();
    ops.push_back(OP_type(bp, state, s, op_type, tau));
    size_t n = ops.size();
    size_t label = sp.size();
    int site;

    for (int i=0; i<s; i++){
      // set_dots(bond[i], 0, i);
      site = bp->operator[](i);
      sp.push_back( Dotv2(sp[site].prev(), site, n-1, i, site));
      sp[sp[site].prev()].set_next(label);
      sp[site].set_prev(label);
      label += 1;
    }
  }
  // //*overload for r value
  // inline void append_ops(OPS& ops, std::vector<int> && bond,  int state, int op_type, double tau){
  //   int s = bond.size();
  //   ops.push_back(OP_type(&bond, state, s, op_type, tau));
  //   for (int i=0; i<s; i++){
  //     set_dots(bond[i], 0, i);
  //   }
  // }

  // //*append to ops
  // inline void append_ops(OPS& ops, const std::vector<int> * const bp,  int state, int op_type, double tau){
  //   int s = bp->size();
  //   ops.push_back(OP_type(bp, state, s, op_type, tau));
  //   for (int i=0; i<s; i++){
  //     set_dots(bp->operator[](i), 0, i);
  //   }
  // }

  // inline void append_ops(OPS& ops, OP_type& op){
  //   append_ops(ops, *op.bond_ptr(), op.state(), op.op_type(), op.tau());
  // }

  //*append to worms
  inline void append_worms(WORMS& wm, int site, int dot_label, double tau){
    wm.push_back(std::make_tuple(site, dot_label, tau));
  }
 
  /*
  *perform one step from given worm.
  If dot is operator then, worm move to exit of the operator. otherwise just assigin spin to dot.
  params
  ------
  int next_dot : next dot.
  int dir : direction worm is moving toward. 1 : move up, 0 : move down.
  int spin : current spin state of worm.
  int site : site worm is at.

  params(member variables)
  ------
  */
  int worm_process_op(size_t& next_dot, size_t& dir, size_t& site, double& wlength, size_t& fl){

    size_t clabel = next_dot;
    auto& dot = spacetime_dots[clabel];

    // ASSERT(site == dot.site(), "site is not consistent");
    if (dot.at_origin()){ //n* if dot is state.
      state[dot.label()] = (state[dot.label()] + fl) % sps; 
      wlength +=1;
      return 0;
    }

    // if (dot.at_worm()){ //n* if dot is at worm
    //   std::get<1>(worms_list[dot.label()]) = spin; // see the definition of WORM
    // }
    size_t sps_prime = sps-1; // = 1 for spin half model

    if (dot.at_operator()){
      size_t dir_in = !dir; //n* direction the worm comes in from the view of operator.
      auto & opstate = ops_main[dot.label()];
      if (opstate.cnt()==0){
        psop.push_back(dot.label());
        pres.push_back(opstate.state());
      }
      opstate.add_cnt();

      if (opstate.cnt() > 10){
        return 1;
      }
      
      // if (dot.label() == 205) {
      //   int gg = 0;
      // }
      wlength += (dir==0) ? -opstate.tau() : opstate.tau();
      size_t size = opstate.size();
      size_t cindex = dot.leg(dir_in, size);
      // opstate.flip_state(cindex);
      opstate.update_state(cindex, fl);
      size_t num = opstate.state();
      int tmp = loperators[opstate.op_type()].markov[num](cindex*(sps_prime) + sps-fl-1, rand_src);

      #ifndef NDEBUG
      int niter = 0;
      // if (dot.label() == 205) {
      //   niter = 10;
      //   int gg = 0;
      // }
      // std::cout << "\n\n" << std::endl;
      for (int i=0; i<niter; i++){
        int tmp_ = loperators[opstate.op_type()].markov[num](cindex*(sps_prime) + sps-fl-1, test_src);
        int nindex_ = tmp_/sps_prime;
        int fl_ = tmp_ % sps_prime + 1;
        printf("test tmp : %d, state : %d\n", tmp_, num ^ (fl_ << (nls*nindex_)));
        
      }
      #endif 
      int nindex = tmp/sps_prime;
      fl = tmp % sps_prime + 1;
      // opstate.flip_state(nindex);
      opstate.update_state(nindex, fl);

      //n* assigin for next step
      dir = nindex/(size);
      wlength += (dir==0) ? opstate.tau() : -opstate.tau();
      site = opstate.bond(nindex%size);
      next_dot = opstate.next_dot(cindex, nindex, clabel);
      return 0;
    }
    return 0;
  }

  /*
  *update worm for W times.
  */
  void worm_update(double& wcount, double& wlength){
    for (WORMS::iterator wsi = worms_list.begin(); wsi != worms_list.end(); ++wsi){
      size_t w_label, site;
      double tau;
      std::tie(site, w_label , tau) = *wsi;
      size_t d_label = w_label;
      auto* dot = &spacetime_dots[d_label];
      double r = uniform(rand_src);
      size_t dir = (size_t)2 * r;//n initial direction is 1.
      size_t ini_dir = dir;
      size_t fl = 1;
      if (nls != 1) fl = static_cast<size_t>((sps-1)*uniform(rand_src)) + 1;
      size_t ini_fl = fl;
      std::copy(state.begin(), state.end(), cstate.begin());
      pres.resize(0);
      psop.resize(0);
      int wl = wlength;
      int br = 0;
      wcount += 1;
      wlength += (dir == 0) ? tau : -tau;
      do{
        d_label = dot->move_next(dir);
        if (worm_process_op(d_label, dir, site, wlength, fl) == 1){
          wlength = wl - ((dir == 0) ? -tau : tau);
          wcount--;
          reset_ops();
          std::copy(cstate.begin(), cstate.end(), state.begin());
          // std::cout << "reset triggered" << std::endl;
          br = 1;
          break;
        }
        if(br==1){std::cout << "what!?" << std::endl;}
        dot = &spacetime_dots[d_label];
      }while((d_label != w_label || ((ini_dir == dir ? -1 : 1)*ini_fl + fl)%sps !=0)&&(br==0)); 
      if(br==1){std::cout << "breakout from loop" << std::endl;}
      
      wlength += (dir == 0) ? -tau : tau;
      check_operators_while_update(w_label, dir ? d_label : dot->prev(), ini_dir, ini_fl, fl, dir);
    }
  }

  /*
  *this function will be called after assigining op_main
  */
  void set_dots(size_t site, size_t dot_type, size_t index){


    size_t label = spacetime_dots.size();

    if (dot_type == -1) {
      ASSERT(label == site, "label must be equal to site");
      spacetime_dots.push_back(
        Dotv2::state(site)
      );
    }else if(dot_type == -2){
      size_t n = worms_list.size();
      spacetime_dots.push_back(
        Dotv2::worm(spacetime_dots[site].prev(), site, n-1, site)
      );
      spacetime_dots[spacetime_dots[site].prev()].set_next(label);
      spacetime_dots[site].set_prev(label);
    }else{
      std::cout << "dot_type must be either -1 or -2" << std::endl;
    }
  }

  /*
  *this function will be called after assigining op_main
  */
  void set_op_dots(size_t site, size_t index){
    size_t label = spacetime_dots.size();
    size_t n = ops_main.size();
    spacetime_dots.push_back(
      Dotv2(spacetime_dots[site].prev(), site, n-1,index, site)
    );
    spacetime_dots[spacetime_dots[site].prev()].set_next(label);
    spacetime_dots[site].set_prev(label);
  }

  /*
  *update given state by given operator ptr;
  */
  static void update_state(typename OPS::iterator opi, STATE& state){
    #ifndef NDEBUG
    auto local_state = opi->get_state_vec();
    STATE state_(opi->size());
    int i=0;
    for (auto x : *(opi->bond_ptr())){
      state_[i] = state[x];
      i++;
    }
    ASSERT(is_same_state(local_state, state_), "the operator can not be applied to the state");
    #endif
    if (opi->is_off_diagonal()) update_state_OD(opi, state);
  }

  /*
  *update given state by given offdiagonal operator ptr;
  */
  static void update_state_OD(typename OPS::iterator opi, STATE& state){
    int index = 0;
    auto const& bond = *(opi->bond_ptr());
    for (auto x : bond){
      state[x] = opi->get_local_state(bond.size() + index);
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
  void check_operators_while_update(int worm_label, int p_label, int ini_dir, int ini_fl, int fl, int dir){
    
    #ifndef NDEBUG
    auto state_ = state;

    int label = 0;
    std::cout << "debug cnt = " << d_cnt << std::endl;
    if (d_cnt == 8){
      int eee;
    }
    d_cnt ++;
    for (const auto& dot:spacetime_dots){


      if (dot.at_operator() && (dot.leg(0, 0)==0)){ //if dot_type is operator
        auto opi = ops_main.begin() + dot.label();
        update_state(opi, state_);
      }
      else if (dot.at_origin()){
        int dot_spin = state[dot.label()];
        ASSERT(state_[dot.site()] == dot_spin, "spin is not consistent");
      }
      // else if (dot.at_worm()){
      //   // int dot_spin = std::get<1>(worms_list[dot.label()]);
      //   // int spin = (worm_label == label) ? (ini_dir^(dot_spin)) : (dot_spin);
      //   // ASSERT(state_[dot.site()] == spin, "spin is not consistent");
      //   // std::cout << "sps : " << sps << std::endl;
      //   if (worm_label == label) state_[dot.site()] = (state_[dot.site()] + ini_fl*(ini_dir ? 1 : -1)) % sps;
      // }

      // if (p_label == label){
      //   state_[dot.site()] = (state_[dot.site()] - fl * (dir ? 1 : -1))%sps;
      // }

      label++;
    }
    ASSERT(is_same_state(state_, state), "operators are not consistent while update worms");
    // std::cout << "hihi" << std::endl;
    #endif 
    return;
  }


  void reset_ops(){
    for (size_t i=0; i<psop.size(); i++){
      ops_main[psop[i]].set_state(pres[i]);
    }
  }


  static bool is_same_state(int n, int m){
    return n==m;
  }

  static bool is_same_state(int n, STATE state){
    int m = state_func::state2num(state, state.size());
    return n==m;
  }

  static bool is_same_state( STATE state_, STATE state){
    int m = state_func::state2num(state, state.size());
    int n= state_func::state2num(state_, state.size());
    return n==m;
  }
  static void printStateAtTime(const STATE& state, double time){
    #ifndef NDEBUG
    std::cout << "current spin at time :" << time << " is : ";
    for (auto spin : state){
      std::cout << spin << " ";
    }
    std::cout << std::endl;
    #else
    return;
    #endif
  }

};






#endif 