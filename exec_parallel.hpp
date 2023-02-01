#pragma once
#include "MainConfig.h"

#include <iostream>
#include <string>
#include <chrono>
#include <type_traits>
#include <string>



#include <bcl.hpp>
#include <libconfig.h++>
#include <worm.hpp>
#include <observable.hpp>
#include <automodel.hpp>
#include <autoobservable.hpp>

#include <alps/alea/batch.hpp>

// batch_obs type is used to store results of observables
typedef alps::alea::batch_acc<double> batch_obs;
typedef alps::alea::batch_result<double> batch_res;

using namespace libconfig;
using namespace std;

// #define DEBUG 1
#define MESTIME 1

#if MESTIME
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;
#endif

extern double elapsed;



// parallel version of exe_worm
template <typename MC>
Worm<MC> exe_worm_parallel(
  model::base_model<MC> spin_model, 
  double T, 
  size_t sweeps, 
  size_t therms, 
  size_t cutoff_l, 
  bool fix_wdensity, 
  int rank,
  std::vector<batch_res>& res, // contains results such as energy, average_sign,, etc
  model::observable obs
){

  // cout << "Hi" << endl;
  using SPINMODEL = model::base_model<MC>;
  if (cutoff_l < 0) cutoff_l = numeric_limits<decltype(cutoff_l)>::max();


  batch_obs ave_sign(1, sweeps); // average sign 
  batch_obs ene(1, sweeps); // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_obs sglt(1, sweeps); 
  batch_obs n_neg_ele(1, sweeps); 
  batch_obs n_ops(1, sweeps); 
  batch_obs N2(1, sweeps); // average of square of number of operators (required for specific heat)
  batch_obs N(1, sweeps); // average of number of operators (required for specific heat)
  batch_obs dH2(1, sweeps); // second derivative by magnetic field
  batch_obs dH(1, sweeps); // first derivative by magnetic field
  
  
  ; // magnetization
  BC::observable M2; // magnetization^2
  BC::observable K; // matnetic susceptibility


  double beta = 1 / T;


  Worm<MC> solver(beta, spin_model, cutoff_l, rank); //template needs for std=14
  // spin_model.lattice.print(std::cout);

  #if MESTIME
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  // double du_time = 0;
  // double wu_time = 0;
  #endif


  int n_kink=0;
  int cnt = 0;
  solver.initStates();
  double wcount = 0;
  double wlength = 0;
  double wdensity = spin_model.Nb;
  for (int i=0; i < therms + sweeps; i++){
    // solver.diagonalUpdate(); 
    solver.diagonalUpdate(wdensity); //n* need to be comment out 
    // printf("%dth iteration\n", i);
    solver.wormUpdate(wcount, wlength);
    // printf("complete Worm update\n");
    if (cnt >= therms){
      int sign = 1;
      // double w_rate = 1;
      double n_neg = 0;
      double n_op = 0;
      double sglt_ = 0;

      double sum_ot = 0; // \sum_{tau} O_{tau} : sum of observables 
      double sum_2_ot = 0; // \sum_{tau} (O_{tau})^2 : sum of square of observables 
      for (const auto&  s : solver.state) {
        if (s==0) sglt_++;
      }
      for (const auto& op : solver.ops_main){
        int sign_ = spin_model.loperators[op.op_type()].signs[op.state()];
        sign *= sign_;
        if (sign_ == -1) n_neg++;
        n_op++;

        // calculate kai (susceptibility)
        double _op_tmp = obs.obs_operators(op.op_type(), op.state());
        // if (_op_tmp != 0) {
        //   cout << op.get_state_vec() << " " << _op_tmp << " " << _op_tmp << endl;
        // }
        sum_ot += _op_tmp;
        sum_2_ot += _op_tmp*_op_tmp;
      }
      double m = (double)solver.ops_main.size();
      double ene_tmp = - m / beta + spin_model.shift();
      N2 << (m * m) * sign;
      N << m * sign;
      ene << ene_tmp * sign;
      ave_sign << sign;
      sglt << sglt_ / spin_model.L;
      n_neg_ele << n_neg;
      n_ops << n_op;
      dH << sum_ot * sign;
      dH2 << (sum_ot*sum_ot - sum_2_ot) * sign;
    }
    if (i <= therms / 2) {
      if (!fix_wdensity){
        if (wcount > 0) wdensity = std::max(spin_model.Nb/ (wlength / wcount), (double)2);
        if (i % (therms / 8 + 1) == 0) {
          wcount /= 2;
          wlength /= 2;
        }
      }
    }
    if (i == therms / 2){
      if (!fix_wdensity && (rank == 0)) std::cout << "Info: average number worms per MCS is reset from " << spin_model.L
                << " to " << wdensity << "(rank=" << rank << ")" <<"\n\n";
      else if (rank == 0) std::cout << "Info: average number worms per MCS is " << wdensity << "(rank=" << rank << ")" <<"\n\n";
    }
    cnt++;
  }

  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
  #endif
  double r = 1-solver.bocnt/ (double)(therms+sweeps); // # of loops breaked out divded by total number of loops.
  // double r_ = 1-r_;




  res.push_back(ave_sign.finalize());
  res.push_back(ene.finalize());
  res.push_back(sglt.finalize());
  res.push_back(n_neg_ele.finalize());
  res.push_back(n_ops.finalize());
  res.push_back(N2.finalize());
  res.push_back(N.finalize());
  res.push_back(dH.finalize());
  res.push_back(dH2.finalize());
  return solver;
}

