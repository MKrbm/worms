// #define RANDOM_SEED 0
#include "MainConfig.h"

#include <iostream>
#include <worm.hpp>
#include <model.hpp>
#include <string>
#include <chrono>
#include <observable.hpp>

#define DEBUG 1
#define MCSTEP 1E5
#define SWEEP 1E4
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
  if (argc < 4) {
    // report version
    std::cout << argv[0] << " Version " << VERSION_MAJOR << "."
              << VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " L J beta" << std::endl;
    return 1;
  }
  std::cout << "MC step : " << MCSTEP << "\n" 
            << "sweep size : " << SWEEP << std::endl;

  int L = std::stoi(argv[1]);
  double J = std::stoi(argv[2]);
  double beta = std::stoi(argv[3]);
  double h = 0;
  BC::observable ene; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  BC::observable umag; // uniform magnetization 
  BC::observable ave_sign; // average sign 


  std::mt19937 rand_src(12345);
  model::heisenberg1D h1(L,h,J);
  worm solver(beta, h1, 4);
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
  solver.ops_sub.resize(0);
  for (int i=0; i < MCSTEP + SWEEP; i++){
    // solver.diagonal_update(); 
    solver.diagonal_update(); //n* need to be comment out 
    solver.check_operators(solver.state, solver.ops_sub);
    solver.check_operators(solver.state, solver.ops_main);
    solver.worm_update();
    solver.swap_oplist();
    if (cnt > SWEEP){
      int sign = 1;
      double mu = 0;
      for (const auto&  s : solver.state) {
        mu += 0.5 - s;
      }
      for (const auto& op : solver.ops_sub){
        std::vector<int> local_state = *op;
        int num = spin_state::state2num(local_state);
        sign *= op->plop->signs[num];
      }
      ene << (- ((double)solver.ops_sub.size()) / beta + h1.shifts[0] * h1.Nb) * sign;
      ave_sign << sign;
      mu /= h1.L;
      umag << mu * sign;
    }
    cnt++;
    // ene << - ((double)solver.ops_sub.size()) / beta + h1.shifts[0] * h1.Nb;
  }


  

  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  cout << "time for diagonal_update : " << du_time/(MCSTEP + SWEEP) << endl
            << "time for worm update : " << wu_time/(MCSTEP+SWEEP) << endl;

  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
  #endif
  std::cout << "Elapsed time = " << elapsed << " sec\n"
            << "Speed = " << (MCSTEP + SWEEP) / elapsed << " MCS/sec\n";
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