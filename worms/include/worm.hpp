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

#include "model.hpp"
#include "BC.hpp"

/* inherit UnionFindTree and add find_and_flip function*/

// template <typename MODEL>

inline int positive_modulo(int i, int n) {
    return (i % n + n) % n;
}
using MODEL = model::heisenberg1D;


class worm{
  public:
  MODEL model;
  double beta;
  int L;
  int W;
  std::vector<int> front_group; // we initilize this by f[i] = -(i+1) so that we can check wether a operator is assigned for the ith site already or not. 
  std::vector<int> front_sides;
  std::vector<std::array<int,2>> end_group;
  std::vector<int> dfront_group;

  using OPS = std::vector<std::array<int, 2>>;
  using OPTAU = std::vector<double>;
  OPS op_tmp1; 
  OPS op_tmp2; 

  OPTAU op_tau_tmp1; 
  OPTAU op_tau_tmp2; 


  OPS& ops_main = op_tmp1; // M x 2 vector (M changes dynamically). first element describe the bond on which the operator act. Second describe the type of operator.
  OPTAU& ops_main_tau = op_tau_tmp1;

  OPS& ops_sub = op_tmp2; // for sub.
  OPTAU& ops_sub_tau = op_tau_tmp2;

  std::vector<std::array<int, 4>> conn_op; // hold the label of connected ops_main. y = x%2, z = x/2, bond[y] is the site to choose. z decide wether go straight or back.
  std::vector<int> state;
  std::vector<std::vector<int>> bonds;

  std::vector<int> worm_start;
  std::vector<double> worm_tau;
  std::vector<int> worm_site;

  decltype(MODEL::trans_prob) trans_prob;


  #ifdef RANDOM_SEED
  std::mt19937 rand_src = std::mt19937(static_cast <unsigned> (time(0)));
  #else
  std::mt19937 rand_src = std::mt19937(2023);
  #endif

  std::uniform_int_distribution<> dist;
  std::uniform_int_distribution<> binary_dice = std::uniform_int_distribution<>(0,1);
  std::uniform_real_distribution<> worm_dist;

  worm(double beta, MODEL model_, int W)
  :L(model.L), beta(beta), model(model_), state(L, 1),
  dist(0,model.Nb-1), worm_dist(0.0, beta), bonds(model.bonds), front_group(L, -1), dfront_group(L),front_sides(L,-1),
  worm_start(W), worm_site(W), worm_tau(W), W(W),
  end_group(L, {-1,-1})
  {
    worm_start.resize(W);
    worm_tau.resize(W);
    worm_site.resize(W);
    #ifdef RANDOM_SEED
    srand(static_cast <unsigned> (time(0)));
    #else
    srand(2023);
    #endif

    for(int i=0; i< L; i++) dfront_group[i] = -(i+1);

    // trans_prob = model.trans_prob = BC::metropolis<decltype(model.weigths)>(model.weigths);
  }
};
#endif 