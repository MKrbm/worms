#pragma once
#include "MainConfig.h"

#include <iostream>
#include <string>
#include <chrono>
#include <type_traits>
#include <string>



#include <bcl.hpp>
#include <libconfig.h++>

#include <worm2.hpp>
#include <observable.hpp>
#include <automodel.hpp>


// #define DEBUG 1
#define MESTIME 1


#if MESTIME
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  using std::chrono::microseconds;

#endif


using namespace libconfig;
using namespace std;
template <typename MC>
std::vector<double> exe_worm(model::base_model<MC> spin_model, const Setting& cfg){

  // cout << "Hi" << endl;
  using SPINMODEL = model::base_model<MC>;
  size_t sweeps, therms, cutoff_l;
  double T;
  bool fix_wdensity = false;
  try
  {
    const Setting& config = cfg["config"];
    sweeps = (long) config.lookup("sweeps");
    therms = (long) config.lookup("therms");
    cutoff_l = (long) config.lookup("cutoff_length");
    T = (float) config.lookup("temperature");
    fix_wdensity = config.lookup("fix_wdensity");

  }
  catch(...)
  {
    cout << "I/O error while reading mc_settings.default settings" << endl;
    cout << "read config file from default instead" << endl;
    const Setting& config = cfg["default"];
    sweeps = (long) config.lookup("sweeps");
    therms = (long) config.lookup("therms");
    cutoff_l = (long) config.lookup("cutoff_length");
    T = (float) config.lookup("temperature");
    fix_wdensity = config.lookup("fix_wdensity");
  }
  if (cutoff_l < 0) cutoff_l = numeric_limits<decltype(cutoff_l)>::max();
  std::cout << "MC step : " << sweeps << "\n" 
          << "thermal size : " << therms << std::endl;

  BC::observable ene; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  BC::observable ave_sign; // average sign 
  BC::observable sglt; 
  BC::observable n_neg_ele; 
  BC::observable n_ops; 
  BC::observable ave_weight; 


  double beta = 1 / T;


  worm<SPINMODEL> solver(beta, spin_model, cutoff_l); //template needs for std=14
  // spin_model.lattice.print(std::cout);


  #if MESTIME
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  double du_time = 0;
  double wu_time = 0;
  #endif


  int n_kink=0;
  int cnt = 0;
  solver.init_states();
  double wcount = 0;
  double wlength = 0;
  double wdensity = spin_model.Nb;
  for (int i=0; i < therms + sweeps; i++){
    // solver.diagonal_update(); 
    solver.diagonal_update(wdensity); //n* need to be comment out 
    // printf("%dth iteration\n", i);
    solver.worm_update(wcount, wlength);
    // printf("complete worm update\n");
    if (cnt >= therms){
      int sign = 1;
      double w_rate = 1;
      double n_neg = 0;
      double n_op = 0;
      double sglt_ = 0;
      for (const auto&  s : solver.state) {
        if (s==0) sglt_++;
      }
      for (const auto& op : solver.ops_main){
        int sign_ = spin_model.loperators[op.op_type()].signs[op.state()];
        sign *= sign_;
        if (sign_ == -1) n_neg++;
        n_op++;
        w_rate *= spin_model.loperators[op.op_type()].ham_rate_vector[op.state()];
        // cout << spin_model.loperators[op.op_type()].ham_rate_vector[op.state()]<< endl;
      }
      double ene_tmp = - (double)solver.ops_main.size() / beta;
      for (int e=0; e<spin_model.N_op; e++){
        ene_tmp += spin_model.shifts[e] * spin_model.bond_t_size[e];
      }
      // ene << ene_tmp * sign;
      ene << ene_tmp * w_rate;
      ave_sign << sign;
      ave_weight << w_rate;
      sglt << sglt_ / spin_model.L;
      n_neg_ele << n_neg;
      n_ops << n_op;
    }
    if (i <= therms / 2) {
      if (!fix_wdensity){
        if (wcount > 0) wdensity = spin_model.Nb/ (wlength / wcount);
        if (i % (therms / 8 + 1) == 0) {
          wcount /= 2;
          wlength /= 2;
        }
      }
    }
    if (i == therms / 2){
      if (!fix_wdensity) std::cout << "Info: average number worms per MCS is reset from " << spin_model.L
                << " to " << wdensity << "\n\n";
      else std::cout << "Info: average number worms per MCS is " << wdensity << "\n\n";
    }
    cnt++;
  }



  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
  #endif

  double r = 1-solver.bocnt/ (double)(therms+sweeps);
  // double r_ = 1-r_;

  std::vector<double> return_value;
  return_value.push_back(ene.mean()/ave_sign.mean());
  return_value.push_back(
    std::sqrt(std::pow(ene.error(r)/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2))
  );
  return_value.push_back(elapsed);
  return_value.push_back((therms+sweeps) / elapsed);



  std::cout << "Total Energy         = "
          << ene.mean()/ave_weight.mean()<< " +- " 
          << std::sqrt(std::pow(ene.error(r)/ave_weight.mean(), 2) + std::pow(ene.mean()/std::pow(ave_weight.mean(),2) * ave_weight.error(r),2))
          << std::endl;

  std::cout << "Elapsed time         = " << elapsed << " sec\n"
            << "Speed                = " << (therms+sweeps) / elapsed << " MCS/sec\n";
  std::cout << "Energy per site      = "
            << ene.mean()/ave_sign.mean() / spin_model.L << " +- " 
            << std::sqrt(std::pow(ene.error(r)/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2)) / spin_model.L
            << std::endl
            << "average sign         = "
            << ave_sign.mean() << " +- " << ave_sign.error(r) << std::endl
            << "average weight rate  = "
            << ave_weight.mean() << " +- " << ave_weight.error(r) << std::endl
            << "dimer operator       = "
            << sglt.mean() << std::endl 
            << "# of operators       = "
            << n_ops.mean() << std::endl
            << "# of neg sign op     = "
            << n_neg_ele.mean() << std::endl
            << "breakout rate        = "
            << 1-r << std::endl;
  return return_value;
}

