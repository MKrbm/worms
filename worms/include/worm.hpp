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

#include <mach-o/dyld.h>

#include <filesystem>
#include <unistd.h>

#include <ctime>
#include <math.h> 

#include "model.hpp"

/* inherit UnionFindTree and add find_and_flip function*/

// template <typename MODEL>
using MODEL = heisenberg1D;


class worm{
  public:
  MODEL model;
  double beta;
  int L;
  int W;
  std::vector<int> front_group;
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
  }

  void init_worm_rand(){
    for (int i=0; i<W; i++){
      worm_site[i] = dist(rand_src);
      worm_tau[i] =  worm_dist(rand_src);
    }
    std::sort(worm_tau.begin(), worm_tau.end(), std::less<double>());
  }

  void init_states(){
    for (auto& x : state){
      x = binary_dice(rand_src) * 2 - 1;
    }
  }

  void init_front_group(){
    front_group = dfront_group;
    std::fill(end_group.begin(), end_group.end(), std::array<int,2>({-1, -1})); 
  }

  void swap_oplist(){
    auto tmp_op = ops_sub;
    auto tmp_tau = ops_sub_tau;
    ops_sub = ops_main;
    ops_sub_tau = ops_main_tau;
    ops_main = tmp_op;
    ops_main_tau = tmp_tau;
  }


  void diagonal_update(){
    double tau_prime = 0;
    double tau = 0;

    double tau_worm = worm_tau[0];
    int n_worm = 0;

    int op_type;
    int N_Dop = model.NDop;
    int N_op = model.Nop;

    int ODtau_label = 0; // label of tau for the last kink at each loop - 1. Next kink label starts from ktau_label

    double r;
    double sum;

    int s0, s1;
    int r_bond; // randomly choosen bond

    const auto &operator_list = model.operator_list;

    init_front_group();
    init_worm_rand();
    ops_main.resize(0);
    ops_main_tau.resize(0);


    while (tau < beta){
      r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
      tau_prime = tau - log(r)/model.rho;

      // put worms on space.
      while ( tau_worm < tau_prime && n_worm <= W){
        worm_start[n_worm] = front_group[worm_site[n_worm]]; //it might be negative value, which will be treeded separately.
        n_worm++;
        tau_worm = worm_tau[n_worm];
      }

      checkODNFlip(ODtau_label, tau_prime);

      r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
      sum = 0;
      int i;
      for(i=0; i<N_Dop; i++){
        sum += model.prob[i];
        if (sum > r) break;
      }
      op_type = i;
      auto& op = operator_list[op_type];

      r_bond = dist(rand_src);
      s0= bonds[r_bond][0];
      s1= bonds[r_bond][1];

      int tmp = op[(state[s0]+1) + (state[s1]+1)/2];
      if ( tmp >= 0){
        // insert is valid.
        ops_main.push_back({r_bond, op_type});
        ops_main_tau.push_back(tau_prime);

        create_conn(ops_main_tau.size()-1, s0, s1);
      }

      tau = tau_prime;
    } //end of while loop

  // connect most fron and most end ops_main
  for (int i=0; i<end_group.size(); i++){
    int op_front = front_group[i];
    int front_side = front_sides[i];
    if (op_front < 0) continue;
    int op_end = end_group[i][0];
    int end_side = end_group[i][1];
    conn_op[op_end][end_side] = op_front;
    conn_op[op_front][2+front_side] = op_end;
  }

  for (auto& x : worm_start){
    if(x<0){
      x = front_group[-(x+1)];
    }
  }
}



int checkODNFlip(int& ODtau_label, double tau_prime){
  /* 
  return tau_label where one start to look for the next time.
  */
  int size = ops_sub.size();
  if (size==0) return 0;

  double ODtau = ops_sub_tau[ODtau_label];
  int bond_label = ops_sub[ODtau_label][0];
  int op_type = ops_sub[ODtau_label][1];
  auto& operator_list = model.operator_list;
  int i = 0, N = 0;

  int s0, s1;
  while((ODtau < tau_prime)&&(ODtau_label <= size-1)){

    if (op_type >= model.NDop){

      s0 = bonds[bond_label][0];
      s1 = bonds[bond_label][1];
      auto& op = operator_list[op_type];

      int tmp = (state[s0]+1) + (state[s1]+1)/2;
      tmp = op[tmp];
      state[s0] = (tmp / 2)*2-1;
      state[s1] = (tmp % 2)*2-1;

      ops_main.push_back({bond_label, op_type});
      ops_main_tau.push_back(ODtau);

      create_conn(ops_main_tau.size()-1, s0, s1); // craete the table of connection.
    }
  
    ODtau_label++;
    i++;
    ODtau = ops_sub_tau[ODtau_label];
    bond_label = ops_sub[ODtau_label][0];
    op_type = ops_sub[ODtau_label][1];
    /* flip state according to off-diagonal operator */
  }
  return i;
}

  void create_conn(int n_op, int s0, int s1){
    auto &lop = front_group[s0];
    auto &lside = front_sides[s0];
    auto &rop = front_group[s1];
    auto &rside = front_sides[s1];

    // if (conn_op.capacity() < n_op) conn_op.resize(2*n_op);
    conn_op.resize(n_op+1);

    if (lop>=0){
      conn_op[lop][2+lside] = n_op;
      conn_op[n_op][0] = lop;
    }else{
      end_group[s0][0] = n_op;
      end_group[s0][1] = 0; //since the site is left side from the operator.
    }

    if (rop>=0){
     conn_op[rop][2+rside] = n_op;
     conn_op[n_op][1] = rop;
    }else{
      end_group[s1][0] = n_op;
      end_group[s1][1] = 1; 
    }

    lop = n_op;
    lside = 0;
    rop = n_op;
    rside = 1;

  }

};



inline std::string getExePath(){
  char buffer[1024];
  uint32_t size = sizeof(buffer);
  std::string path;

#if defined (WIN32)
  GetModuleFileName(nullptr,buffer,size);
  path = buffer;
#elif defined (__APPLE__)
  namespace fs = std::__fs::filesystem;
  if(_NSGetExecutablePath(buffer, &size) == 0)
  {
  path = buffer;
  }
#elif defined(UNIX) || defined(unix) || defined(__unix) || defined(__unix__)
  throw std::runtime_error("UNIX is not available \n");
  exit;
  int byte_count = readlink("/proc/self/exe", buffer, size);
  if (byte_count  != -1)
  {
  path = std::string(buffer, byte_count);
  }
#endif

  fs::path p = path;
  return static_cast<std::string>(p.parent_path());
}

#endif 