#pragma once
#include "MainConfig.h"
#include "options.hpp"

#include <iostream>
#include <worm.hpp>


#include <heisenberg.hpp>
#include <Shastry.hpp>
#include <ladder.hpp>
#include <MG.hpp>



#include <testmodel.hpp>
#include <operator.hpp>
#include <string>
#include <chrono>
#include <observable.hpp>
#include <lattice/graph.hpp>
#include <lattice/coloring.hpp>
#include <bcl.hpp>
#include <type_traits>
#include <string>


// #define DEBUG 1
#define MESTIME 1


#if MESTIME
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  using std::chrono::microseconds;

#endif

template <class, template <class> class>
struct is_instance : public std::false_type {};

template <class T, template <class> class U>
struct is_instance<U<T>, U> : public std::true_type {};


template <typename SPINMODEL>
std::vector<double> exe_worm(SPINMODEL spin_model, options* opt_ptr,
  typename std::enable_if<(is_instance<SPINMODEL,model::heisenberg>::value||is_instance<SPINMODEL,model::heisenberg_v2>::value),std::nullptr_t>::type = nullptr){

  // std::cout << "test L : " << opt_ptr -> sweeps << std::endl;

  auto opt = *opt_ptr;
  std::cout << "MC step : " << opt.sweeps << "\n" 
          << "thermal size : " << opt.therm << std::endl;

  BC::observable ene; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  BC::observable umag; // uniform magnetization 
  BC::observable ave_sign; // average sign 
  BC::observable umag2;
  double beta = 1 / opt.T;
  size_t co = opt.co;
  bool fix_wdensity = opt.fix_wdensity;


  Worm<SPINMODEL> solver(beta, spin_model, co); //template needs for std=14
  // std::vector<std::vector<int>> states;
  spin_model.lattice.print(std::cout);


  #if MESTIME
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  double du_time = 0;
  double wu_time = 0;
  #endif


  int n_kink=0;
  int cnt = 0;
  solver.initStates();
  double wcount = 0;
  double wlength = 0;
  double wdensity = spin_model.lattice.num_bonds();
  double wdty = opt_ptr->wdty;
  for (int i=0; i < opt.therm + opt.sweeps; i++){
    // solver.diagonalUpdate(); 
    solver.diagonalUpdate(wdensity); //n* need to be comment out 
    solver.WormUpdate(wcount, wlength);
    if (cnt >= opt.therm){
      int sign = 1;
      double mu = 0;
      for (const auto&  s : solver.state) {
        mu += 0.5 - s;
      }
      for (const auto& op : solver.ops_main){
        sign *= spin_model.loperators[op.op_type()].signs[op.state()];
      }
      double ene_tmp = - (double)solver.ops_main.size() / beta;
      for (int e=0; e<spin_model.Nop; e++){
        ene_tmp += spin_model.shifts[e] * spin_model.bond_t_size[e];
      }
      ene << ene_tmp * sign;
      ave_sign << sign;
      mu /= spin_model.L;
      umag << mu * sign;
      umag2 << mu * mu * sign;
    }
    if (i <= opt.therm / 2) {
      if (!fix_wdensity){
        if (wcount > 0) wdensity = spin_model.lattice.num_bonds()/ (wlength / wcount);
        if (i % (opt.therm / 8 + 1) == 0) {
          wcount /= 2;
          wlength /= 2;
        }
      }
    }
    if (i == opt.therm / 2){
      if (!fix_wdensity) std::cout << "Info: average number worms per MCS is reset from " << spin_model.L
                << " to " << wdensity << "\n\n";
      else std::cout << "Info: average number worms per MCS is " << wdensity << "\n\n";
    }
    cnt++;
  }


  
  double r = 1-solver.bocnt/ (double)(opt.therm+opt.sweeps);

  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
  #endif

  std::vector<double> return_value;
  return_value.push_back(ene.mean()/ave_sign.mean());
  return_value.push_back(
    std::sqrt(std::pow(ene.error(r)/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2))
  );
  return_value.push_back(elapsed);
  return_value.push_back((opt.therm+opt.sweeps) / elapsed);



  std::cout << "Total Energy         = "
          << ene.mean()/ave_sign.mean()<< " +- " 
          << std::sqrt(std::pow(ene.error(r)/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2))
          << std::endl;

  std::cout << "Elapsed time         = " << elapsed << " sec\n"
            << "Speed                = " << (opt.therm+opt.sweeps) / elapsed << " MCS/sec\n";
  std::cout << "Energy per site      = "
            << ene.mean()/ave_sign.mean() / spin_model.lattice.num_sites() << " +- " 
            << std::sqrt(std::pow(ene.error(r)/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2)) / spin_model.lattice.num_sites()
            << std::endl
            << "average sign              = "
            << ave_sign.mean() << " +- " << ave_sign.error(r) << std::endl
            << "Uniform Magnetization     = "
            << umag.mean()/ave_sign.mean() << " +- " 
            << std::sqrt(std::pow(umag.error(r)/ave_sign.mean(), 2) + std::pow(umag.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2))
            << std::endl
            << "Uniform Magnetization^2   = "
            << umag2.mean()/ave_sign.mean() << " +- "
            << std::sqrt(std::pow(umag2.error(r)/ave_sign.mean(), 2) + std::pow(umag2.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2))
            << std::endl            
            << "breakout rate             = "
            << 1-r << std::endl;
  return return_value;
}


template <typename SPINMODEL>
std::vector<double> exe_worm(SPINMODEL spin_model, options* opt_ptr,
  typename std::enable_if<!(is_instance<SPINMODEL,model::heisenberg>::value||is_instance<SPINMODEL,model::heisenberg_v2>::value),std::nullptr_t>::type = nullptr){


  std::cout << "this is called" << std::endl;

  auto opt = *opt_ptr;
  std::cout << "MC step : " << opt.sweeps << "\n" 
          << "thermal size : " << opt.therm << std::endl;

  BC::observable ene; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  BC::observable ave_sign; // average sign 
  BC::observable sglt; 
  BC::observable n_neg_ele; 
  BC::observable n_ops; 
  BC::observable ave_weight; 
  bool fix_wdensity = opt.fix_wdensity;


  double beta = 1 / opt.T;
  size_t co = opt.co;


  Worm<SPINMODEL> solver(beta, spin_model, co); //template needs for std=14
  // std::vector<std::vector<int>> states;
  spin_model.lattice.print(std::cout);


  #if MESTIME
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  double du_time = 0;
  double wu_time = 0;
  #endif


  int n_kink=0;
  int cnt = 0;
  solver.initStates();
  double wcount = 0;
  double wlength = 0;
  double wdensity = spin_model.lattice.num_bonds();
  double wdty = opt_ptr->wdty;
  for (int i=0; i < opt.therm + opt.sweeps; i++){
    // solver.diagonalUpdate(); 
    solver.diagonalUpdate(wdensity); //n* need to be comment out 
    // printf("%dth iteration\n", i);
    solver.WormUpdate(wcount, wlength);
    // printf("complete Worm update\n");
    if (cnt >= opt.therm){
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
      for (int e=0; e<spin_model.Nop; e++){
        ene_tmp += spin_model.shifts[e] * spin_model.bond_t_size[e];
      }
      // ene << ene_tmp * sign;
      ene << ene_tmp * w_rate;
      ave_sign << sign;
      ave_weight << w_rate;
      sglt << sglt_ / spin_model.lattice.num_sites();
      n_neg_ele << n_neg;
      n_ops << n_op;
    }
    if (i <= opt.therm / 2) {
      if (!fix_wdensity){
        if (wcount > 0) wdensity = spin_model.lattice.num_bonds()/ (wlength / wcount);
        if (i % (opt.therm / 8 + 1) == 0) {
          wcount /= 2;
          wlength /= 2;
        }
      }
    }
    if (i == opt.therm / 2){
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

  double r = 1-solver.bocnt/ (double)(opt.therm+opt.sweeps);
  // double r_ = 1-r_;

  std::vector<double> return_value;
  return_value.push_back(ene.mean()/ave_sign.mean());
  return_value.push_back(
    std::sqrt(std::pow(ene.error(r)/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2))
  );
  return_value.push_back(elapsed);
  return_value.push_back((opt.therm+opt.sweeps) / elapsed);



  std::cout << "Total Energy         = "
          << ene.mean()/ave_weight.mean()<< " +- " 
          << std::sqrt(std::pow(ene.error(r)/ave_weight.mean(), 2) + std::pow(ene.mean()/std::pow(ave_weight.mean(),2) * ave_weight.error(r),2))
          << std::endl;

  std::cout << "Elapsed time         = " << elapsed << " sec\n"
            << "Speed                = " << (opt.therm+opt.sweeps) / elapsed << " MCS/sec\n";
  std::cout << "Energy per site      = "
            << ene.mean()/ave_sign.mean() / spin_model.lattice.num_sites() << " +- " 
            << std::sqrt(std::pow(ene.error(r)/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(r),2)) / spin_model.lattice.num_sites()
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