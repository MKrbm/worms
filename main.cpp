// #define RANDOM_SEED 0
#include "MainConfig.h"
#include "options.hpp"

#include <iostream>
#include <worm.hpp>
#include <model.hpp>
#include <string>
#include <chrono>
#include <observable.hpp>

#define DEBUG 1
#define MESTIME 1

#if MESTIME
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  using std::chrono::microseconds;

#endif

int main(int argc, char* argv[])
{

  options opt(argc, argv, 16, 1.0);
  if (!opt.valid) std::exit(-1);
  double beta = 1 / opt.T;
  int L = opt.L;
  double J = 1;
  double h = opt.H;
  // double beta = std::stoi(argv[3]);

  std::cout << "MC step : " << opt.sweeps << "\n" 
            << "thermal size : " << opt.therm << std::endl;

  BC::observable ene; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  BC::observable umag; // uniform magnetization 
  BC::observable ave_sign; // average sign 


  // std::mt19937 rand_src(12345);
  model::heisenberg1D h1(L,h,J);
  worm solver(beta, h1);
  // std::vector<std::vector<int>> states;


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
  int spin = 1;
  for (auto& s : solver.state){
    s = spin;
    spin^=1;
  }
  for (int i=0; i < opt.therm + opt.sweeps; i++){
    // solver.diagonal_update(); 
    solver.diagonal_update(3); //n* need to be comment out 
    solver.worm_update();
    if (cnt >= opt.therm){
      int sign = 1;
      double mu = 0;
      for (const auto&  s : solver.state) {
        mu += 0.5 - s;
      }
      for (const auto& op : solver.ops_main){
        sign *= h1.loperators[op.op_type()].signs[op.state()];
      }
      ene << (- ((double)solver.ops_main.size()) / beta + h1.shifts[0] * h1.Nb) * sign;
      ave_sign << sign;
      mu /= h1.L;
      umag << mu * sign;
    }
    cnt++;
  }


  

  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  cout << "time for diagonal_update : " << du_time/(opt.therm+opt.sweeps) << endl
            << "time for worm update : " << wu_time/(opt.therm+opt.sweeps) << endl;

  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
  #endif
  std::cout << "Elapsed time = " << elapsed << " sec\n"
            << "Speed = " << (opt.therm+opt.sweeps) / elapsed << " MCS/sec\n";
  std::cout << "Energy             = "
            << ene.mean()/ave_sign.mean() / h1.L << " +- " 
            << std::sqrt(std::pow(ene.error()/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2)) / h1.L
            << std::endl
            << "Uniform Magnetization     = "
            << umag.mean()/ave_sign.mean() << " +- " 
            << std::sqrt(std::pow(umag.error()/ave_sign.mean(), 2) + std::pow(umag.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2))
            << std::endl
            << "average sign     = "
            << ave_sign.mean() << " +- " << ave_sign.error() << std::endl;
}