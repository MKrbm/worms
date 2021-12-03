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
using MODEL = heisenberg1D;


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

    trans_prob = model.trans_prob = metropolis<decltype(model.weigths)>(model.weigths);
  }

  void init_worm_rand(){
    for (int i=0; i<W; i++){
      worm_site[i] = dist(rand_src);
      worm_tau[i] =  worm_dist(rand_src);
    }
    std::sort(worm_tau.begin(), worm_tau.end(), std::less<double>());
    std::fill(worm_start.begin(), worm_start.end(), 0); 
  }

  void init_states(){
    for (auto& x : state){
      x = binary_dice(rand_src) * 2 - 1;
    }
  }

  void init_front_group(){
    front_group = dfront_group;
    std::fill(end_group.begin(), end_group.end(), std::array<int,2>({-1, -1})); 
    std::fill(front_sides.begin(), front_sides.end(), -1); 
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
      while ( tau_worm < tau_prime && n_worm < W){
        worm_start[n_worm] = front_group[worm_site[n_worm]]; //it might be negative value, which will be treeded separately.
        n_worm++;
        tau_worm = worm_tau[n_worm];
      }

      checkODNFlip(ODtau_label, tau_prime);

      
      // sum = 0;
      // int i;
      // for(i=0; i<N_Dop; i++){
      //   sum += model.prob[i];
      //   if (sum > r) break;
      // }
      // op_type = i;
      // op_type = model.DopAtRand(r);
      op_type = chooseAtRand(model.prob);

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


  /* 

  check over off-diagonal operators inside sub operators list (ops_sub) before given tau_prime.
  If operator is not an off-diagonal ops, then it ignore and continue to the next ODtau_label.
  If operator is an off-diagonal, then, we append to ops_main and update states accordingly.

  params
  ------
  int ODtau_label : the label for sub operators (ops_sub) our search starts from.
  double tau_prime : tau_prime

  return 
  ------
  int : tau_label where one start to look for the next time.
  */
  int checkODNFlip(int& ODtau_label, double tau_prime){

    int size = ops_sub.size();
    if (size==0 || ODtau_label >= size) return 0;

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


  /*
  update front_group and end_group. and create a connection for given n_op

  params
  ------

  int n_op : operator label that will assigned as a new label. 
  int s0 : one of the site the operator acts on
  int s1 : "
  
  */
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
  /*
  perform one step from given worm.
  params
  ------
  int[] state : current state
  int dir : from 0 to 3 that describes the direction (rule is the same as connection). y = dir%2, z = dir/2, bond[y] is the site. z decide wether go up or down.
  int op_label : the last operator label.
  params(member variables)
  ------
  worm_dir : hold the direction to move.
  trans_prob : holds probability that the give op_type transition to others (including itself).

  */
  void worm_step(std::vector<int>& CState, int& dir, int& op_label, int& site){
    int op_label_ = conn_op[op_label][dir]; // candidate op_label from current op_label.
    int bond_label = ops_main[op_label_][0];
    auto bond = bonds[bond_label];
    int LorR = (site != bond[0]); // 0 means left side, 1 means right side
    int UorD = dir/2; // 0 means the worm trave in a down direction, 1 means up
    int& type = ops_main[op_label_][1];
    const auto& worm_dir = model.worm_dir[type];
    const auto& prob = trans_prob[type];

    int trans_type = chooseAtRand(prob);
    int reldir = worm_dir[trans_type]; //reldir specify the relative direction from the pov of worm. check the definitoin of model.worm_dir

    if (reldir/2 == 1) CState[site] *= -1;
    CState = getCState(op_label+1, op_label_-1, CState);
    dir = 2 * ((UorD + reldir/2 + 1) % 2) + (LorR + reldir%2) % 2;
    type = trans_type;
    op_label = op_label_;
    site = bond[dir%2];
    CState[site] *= -1;
  }

  template <typename PROB>
  int chooseAtRand(const PROB& prob){
      double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
      double sum = 0;
      int i;
      for(i=0; i<prob.size()-1; i++){
          sum += prob[i];
          if (sum >= r) break;
      }

      return i;
  }

  void worm_update(){
    
    int s_label = 0;
    auto index = sortindex(worm_start);
    auto tmp_start = worm_start;
    auto tmp_site = worm_site;
    int label;

    for (int i=0; i<index.size(); i++){
      worm_start[i] = tmp_start[index[i]];
      worm_site[i] = tmp_site[index[i]];
    }

    std::vector<int> CState = state;
    for(int j=0; j<W; j++){
      int site = worm_site[j];
      label = worm_start[j];
      std::vector<int> bond = bonds[ops_main[label][j]];
      int i;
      for (i=0; i<2; i++){
          if (bond[i]==site) break;
      }
      

      CState = getCState(s_label, label, CState);
      CState[site] *= -1;
      int dir = 2 + i;
      int ori_dir = dir;
      int ori_label = label;
      do{
          worm_step(CState, dir, label, site);
      }while((ori_dir!=dir)||(ori_label!=label));
      s_label = ori_label+1;

      CState = getCState(s_label, ops_main.size()+s_label-1, CState);

    }

  }

  /* update state from start to end (includes both label)
  */

  std::vector<int> getCState(int start, int end, std::vector<int> Cstate){
    if (end<start) end+=ops_main.size();
    for(int i=0; i<end-start+1; i++){
      int op_label = (i + start) % ops_main.size();
      int bond_label = ops_main[op_label][0];
      int type = ops_main[op_label][1];
      if (type<model.NDop) continue;
      int s0 = bonds[bond_label][0];
      int s1 = bonds[bond_label][1];
      auto& op = model.operator_list[type];

      int tmp = (Cstate[s0]+1) + (Cstate[s1]+1)/2;
      tmp = op[tmp];
      Cstate[s0] = (tmp / 2)*2-1;
      Cstate[s1] = (tmp % 2)*2-1;
    }
    return Cstate;
  }

  std::vector<int> sortindex(std::vector<int> vec){
    std::vector<int> index(vec.size());
    for(int i=0; i < vec.size(); i++) index[i] = i;

    std::sort(index.begin(), index.end(), [&](int a, int b) {return vec[a] < vec[b];} );

    return index;
  }

};


// inline std::string getExePath(){
//   char buffer[1024];
//   uint32_t size = sizeof(buffer);
//   std::string path;

  

// #if defined (WIN32)
//   GetModuleFileName(nullptr,buffer,size);
//   path = buffer;
// #elif defined (__APPLE__)
//   namespace fs = std::__fs::filesystem;
//   if(_NSGetExecutablePath(buffer, &size) == 0)
//   {
//   path = buffer;
//   }
//   fs::path p = path;
  
// #elif defined(UNIX) || defined(unix) || defined(__unix) || defined(__unix__)
//   // throw std::runtime_error("UNIX is not available \n");
//   exit;
//   int byte_count = readlink("/proc/self/exe", buffer, size);
//   if (byte_count  != -1)
//   {
//   path = std::string(buffer, byte_count);
//   }
//   std::filesystem::path p = path;
// #endif

//   return static_cast<std::string>(p.parent_path());
  
// }

#endif 