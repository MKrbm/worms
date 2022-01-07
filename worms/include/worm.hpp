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


using spin_state::Operatorv2;
using spin_state::Dotv2;

// using MODEL = model::heisenberg1D;
using OPS = std::vector<Operatorv2>;
using STATE = model::STATE;
using BOND = model::BOND;
using WORMS = spin_state::WORM_ARR;
using DOTS = std::vector<Dotv2>;
using size_t = std::size_t;


template <typename MODEL>
class worm{
  public:
  MODEL model;
  double beta;
  int L;
  OPS ops_main; //contains operators.
  OPS ops_sub; // for sub.
  STATE state;
  STATE cstate;
  DOTS spacetime_dots; //contain dots in space-time.
  WORMS worms_list;

  std::vector< BOND > bonds;
  std::vector<size_t> bond_type;
  typedef typename MODEL::base_spin_model base_spin_model;

  // const spin_state::state_func func = {2};
  typedef spin_state::state_func<base_spin_model::S> state_func;

  //declaration for random number generator
  // typedef model::local_operator::engine_type engine_type;
  typedef std::mt19937 engine_type;
  #ifdef RANDOM_SEED
  engine_type rand_src = engine_type(static_cast <unsigned> (time(0)));
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
  bonds(model.bonds),bond_type(model.bond_type) ,state(L),cstate(L),
  loperators(model.loperators), leg_sizes(model.leg_size),
  lop(loperators[0]),markov(lop.markov)
  {
    cout << "beta : " << beta << endl;
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
    x = 0;
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
    ops_sub.push_back(Operatorv2::sentinel(1)); //*sentinels
    double tau = expdist(rand_src);
    for (OPS::iterator opi = ops_sub.begin(); opi != ops_sub.end();){
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
            append_ops(ops_main, spacetime_dots, &bond, (u<<bond.size()) | u, lop_label, tau);

            if (lop_label == 1) {
              int  x = 1;
            }
            //*append ops
            // size_t s = bond.size();
            // ops_main.push_back(Operatorv2(&bond, (u<<s) | u, s, lop_label, tau));
            // size_t n = ops_main.size();
            // size_t label = spacetime_dots.size();
            // size_t site;

            // for (size_t i=0; i<s; i++){
            //   site = bond[i];
            //   spacetime_dots.push_back( Dotv2(spacetime_dots[site].prev(), site, n-1,i, site));
            //   spacetime_dots[spacetime_dots[site].prev()].set_next(label);
            //   spacetime_dots[site].set_prev(label);
            //   label += 1;
            // }
            //* end
          }
        }
        tau += expdist(rand_src);
      }else{ //*if tau went over the operator time.
        if (opi->is_off_diagonal()) {
          // auto const& bond = *opi->bond_ptr();
          update_state(opi, cstate);
          append_ops(ops_main, spacetime_dots, opi->bond_ptr(), opi->state(), opi->op_type(),opi->tau());
          //* append ops
          // size_t size = opi->size();
          // ops_main.push_back(Operatorv2(opi->bond_ptr(), opi->state(), size, opi->op_type(), opi->tau()));
          // size_t n = ops_main.size();
          // size_t label = spacetime_dots.size();
          // size_t site;
          // for (size_t i=0; i<size; i++){
          //   site = opi->bond(i);
          //   spacetime_dots.push_back( Dotv2(spacetime_dots[site].prev(), site, n-1,i, site));
          //   spacetime_dots[spacetime_dots[site].prev()].set_next(label);
          //   spacetime_dots[site].set_prev(label);
          //   label += 1;
          // }
          //* end append
          printStateAtTime(cstate, tau);
        }
        ++opi;
      }
    } //end of while loop
  }

  // //*append to ops
  static void append_ops(OPS& ops, DOTS& sp, const BOND * const bp, int state, int op_type, double tau){

    int s = bp->size();
    ops.push_back(Operatorv2(bp, state, s, op_type, tau));
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
  //   ops.push_back(Operatorv2(&bond, state, s, op_type, tau));
  //   for (int i=0; i<s; i++){
  //     set_dots(bond[i], 0, i);
  //   }
  // }

  // //*append to ops
  // inline void append_ops(OPS& ops, const std::vector<int> * const bp,  int state, int op_type, double tau){
  //   int s = bp->size();
  //   ops.push_back(Operatorv2(bp, state, s, op_type, tau));
  //   for (int i=0; i<s; i++){
  //     set_dots(bp->operator[](i), 0, i);
  //   }
  // }

  // inline void append_ops(OPS& ops, Operatorv2& op){
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
  void worm_process_op(size_t& next_dot, size_t& dir, size_t& site, double& wlength){

    size_t clabel = next_dot;
    auto& dot = spacetime_dots[clabel];

    // ASSERT(site == dot.site(), "site is not consistent");
    if (dot.at_origin()){ //n* if dot is state.
      state[dot.label()] ^= 1; 
      wlength +=1;
      return;
    }

    // if (dot.at_worm()){ //n* if dot is at worm
    //   std::get<1>(worms_list[dot.label()]) = spin; // see the definition of WORM
    // }

    if (dot.at_operator()){
      size_t dir_in = !dir; //n* direction the worm comes in from the view of operator.
      auto & opstate = ops_main[dot.label()];
      wlength += (dir==0) ? -opstate.tau() : opstate.tau();
      size_t size = opstate.size();
      size_t cindex = dot.leg(dir_in, size);
      opstate.flip_state(cindex);
      size_t num = opstate.state();
      // double r = uni_dist(rand_src);
      int nindex = loperators[opstate.op_type()].markov[num](cindex, rand_src);
      // size_t nindex = markov[num](cindex, rand_src);
      opstate.flip_state(nindex);

      //n* assigin for next step
      dir = nindex/(size);
      wlength += (dir==0) ? opstate.tau() : -opstate.tau();
      site = opstate.bond(nindex%size);
      next_dot = opstate.next_dot(cindex, nindex, clabel);
      return;
    }
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
      wcount += 1;
      wlength += (dir == 0) ? tau : -tau;
      do{
        check_operators_while_update(w_label, dir ? d_label : dot->prev(), ini_dir);
        d_label = dot->move_next(dir);
        worm_process_op(d_label, dir, site, wlength);
        dot = &spacetime_dots[d_label];
      }while(d_label != w_label); 
      wlength += (dir == 0) ? -tau : tau;
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
  static void update_state(OPS::iterator opi, STATE& state){
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
  static void update_state_OD(OPS::iterator opi, STATE& state){
    int index = 0;
    auto const& bond = *(opi->bond_ptr());
    for (auto x : bond){
      state[x] = opi->get_spin(bond.size() + index);
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
  void check_operators_while_update(int worm_label, int p_label, int ini_dir){
    
    #ifndef NDEBUG
    auto state_ = state;

    int label = 0;
    for (const auto& dot:spacetime_dots){


      if (dot.at_operator() && (dot.leg(0, 0)==0)){ //if dot_type is operator
        auto opi = ops_main.begin() + dot.label();
        update_state(opi, state_);
      }
      else if (dot.at_origin()){
        int dot_spin = state[dot.label()];
        ASSERT(state_[dot.site()] == dot_spin, "spin is not consistent");
      }
      else if (dot.at_worm()){
        // int dot_spin = std::get<1>(worms_list[dot.label()]);
        // int spin = (worm_label == label) ? (ini_dir^(dot_spin)) : (dot_spin);
        // ASSERT(state_[dot.site()] == spin, "spin is not consistent");
        if (worm_label == label) state_[dot.site()] = 1^state_[dot.site()];
      }

      if (p_label == label){
        state_[dot.site()] = 1^state_[dot.site()];
      }

      label++;
    }
    ASSERT(is_same_state(state_, state), "operators are not consistent while update worms");
    // std::cout << "hihi" << std::endl;
    #endif 
    return;
  }


  static bool is_same_state(int n, int m){
    return n==m;
  }

  static bool is_same_state(int n, STATE state){
    int m = spin_state::state2num(state, state.size());
    return n==m;
  }

  static bool is_same_state( STATE state_, STATE state){
    int m = spin_state::state2num(state, state.size());
    int n= spin_state::state2num(state_, state.size());
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