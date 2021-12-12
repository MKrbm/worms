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

/* inherit UnionFindTree and add find_and_flip function*/

// template <typename MODEL>

inline int positive_modulo(int i, int n) {
    return (i % n + n) % n;
}
using MODEL = model::heisenberg1D;
using OPS = std::vector<model::OpStatePtr>;
using BSTATE = model::BottomState;
using WORMS = model::Worms;
using DOTS = std::vector<model::Dot>;




class worm{
  public:
  MODEL model;
  double beta;
  int L;
  int W; //number of worms
  std::vector<int> front_dots;// initilize this by f[i] = -(i+1) so that we can check wether a operator is assigned for the ith site already
  std::vector<int> end_dots; //initialize in the same manner.
  OPS ops_tmp1; 
  OPS ops_tmp2; 


  OPS& ops_main = ops_tmp1;  //contains operators.
  OPS& ops_sub = ops_tmp2; // for sub.
  model::BStatePtr pstate;
  model::WormsPtr pworms;

  model::Worms& worms = *pworms;
  model::BottomState& state = *pstate;


  // BSTATE& state;
  // WORMS& worms;

  DOTS spacetime_dots; //contain dots in space-time.

  std::vector<double> worms_tau;
  std::vector<std::vector<int>> bonds;

  //declaration for random number generator

  #ifdef RANDOM_SEED
  std::mt19937 rand_src = std::mt19937(static_cast <unsigned> (time(0)));
  #else
  std::mt19937 rand_src = std::mt19937(2023);
  #endif

  //for choosing bonds
  std::uniform_int_distribution<> dist;

  // for binary dice
  std::uniform_int_distribution<> binary_dice = std::uniform_int_distribution<>(0,1); 

  // random distribution from 0 to beta 
  std::uniform_real_distribution<> worm_dist;


  // reference of member variables from model class

  static const int N_op = MODEL::Nop;

  std::array<model::local_operator, N_op>& loperators; //holds multiple local operators
  std::array<int, N_op>& leg_sizes; //leg size of local operators;
  std::array<double, N_op>& operator_cum_weights;



  worm(double beta, MODEL model_, int W)
  :model(model_), L(model.L), beta(beta), W(W),
  dist(0, model.Nb-1), worm_dist(0.0, beta),
  bonds(model.bonds), 
  pstate(model::BStatePtr(new BSTATE(L))), pworms(model::WormsPtr(new WORMS(W))),
  loperators(model.loperators), leg_sizes(model.leg_size),
  operator_cum_weights(model.operator_cum_weights), worms_tau(W),
  front_dots(L,0), end_dots(L,0)
  {
    #ifdef RANDOM_SEED
    srand(static_cast <unsigned> (time(0)));
    #else
    srand(2023);
    #endif
    worms_tau.resize(W);
    // pstate = new BSTATE(L);
    // pworms = new WORMS(W);
    
  }

  // functions for initializing

  /* initialize worms */
  void init_worms_rand(){
    for (int i=0; i<W; i++){
      // pworms->worm_site[i] = dist(rand_src);
      // pworms->tau_list[i] =  worm_dist(rand_src);

      worms.worm_site[i] = dist(rand_src);
      worms.tau_list[i] =  worm_dist(rand_src);
    }
    std::sort(worms.tau_list.begin(), worms.tau_list.end(), std::less<double>());
  }

  /* 
  initialize front and end groups 
  */

  void init_front_n_end(){
    for (int i=0; i<L; i++){
      front_dots[i] = -(i+1);
      end_dots[i] = -(i+1);
    }
  }

  void init_states(){
  for (auto& x : state){
    x = binary_dice(rand_src);
    }
  }


  void swap_oplist(){
    auto tmp_op = ops_sub;
    ops_sub = ops_main;
    ops_main = tmp_op;
  }

  void diagonal_update(){
    double tau_prime = 0;
    double tau = 0;
    int n_worm = 0;
    init_front_n_end();
    init_worms_rand();


    auto& worm_site = worms.worm_site;
    auto& worm_tau_list = worms.tau_list;
    double worm_tau = worm_tau_list[0];
    int N_op = model.Nop;


    int optau = 0;
    if (ops_sub.size()) optau = ops_sub[0]->tau;

    double sum;

    int s0, s1;
    int r_bond; // randomly choosen bond



    ops_main.resize(0);


    //set worms
    while (true){
      // cout << "hi" << endl;
      double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
      tau_prime = tau - log(r)/model.rho;

      // put worms on space.
      while(worm_tau<tau_prime && n_worm < W){
        int site = worm_site[n_worm];
        spacetime_dots.emplace_back(
          site, worm_tau, front_dots[site], worms.data() + n_worm,
          pworms, 2
        );
        worms[n_worm] = state[site];
        n_worm++;
        worm_tau = worm_tau_list[n_worm];
        setfrontNend(site, spacetime_dots.size()-1);
      }

      checkODNFlip(optau, tau_prime);

      //choose and insert diagonal operator.

      if (tau_prime > beta) break;

      r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
      double max_ = *(operator_cum_weights.end()-1);
      int lop_label;
      for(lop_label=0; lop_label < N_op; lop_label++){
        if (operator_cum_weights[lop_label] >= r * max_) break;
      }

      int leg_size = leg_sizes[lop_label]; //size of choosen operator
      auto& lop = loperators[lop_label];
      auto& diag_cum_weight = lop.diagonal_cum_weight;
      max_ = lop.total_weights;

      r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
      int s_num; //choose local state 
      for (s_num=0; s_num < (1<<leg_size); s_num++){
        if (diag_cum_weight[s_num] >= r * max_) break;
      }

      // choose bond
      r_bond = dist(rand_src);
      int tuggle = 1;
      auto local_state = model::num2state(s_num + (s_num<<leg_size ), 2*leg_size);
      std::vector<int> labels(leg_size);
      auto bond = bonds[r_bond];

      int n_dots = spacetime_dots.size();
      for (int i=0; i<leg_size; i++){
        labels[i] = n_dots;
        n_dots++;
        int s = bond[i];
        if (state[s] != local_state[i]) tuggle = 0;
      }



      if ( tuggle ){
        ops_main.emplace_back(
            new model::OpState(
              local_state,
              &lop,
              labels,
              bond,
              tau_prime
          )
        );

        int dot_label = spacetime_dots.size();
        int n = ops_main.size();
        for (int i=0; i<leg_size; i++){
          spacetime_dots.emplace_back(
            bond[i], tau_prime, front_dots[bond[i]], ops_main[n-1]->data() + i,
            ops_main[n-1], 1
          );
          setfrontNend(bond[i], n_dots);
          dot_label++;
        }
      }
      
      tau = tau_prime;

    } //end of while loop
    

  }

  void checkODNFlip(int& optau, int tau_prime){
    return;
  }

  void setfrontNend(int site; int label){
    if (end_dots[site] < 0) end_dots[site] = label;
    front_dots[site] = label;
  }
};

#endif 