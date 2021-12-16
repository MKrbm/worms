#define RANDOM_SEED 1

#include <iostream>
#include <worm.hpp>
#include <model.hpp>
#include <string>
#include <chrono>

#include "MainConfig.h"
#define DEBUG 1
#define MCSTEP 50000
#define SWEEP 50
#define MESTIME 1


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
  double h = 1;

  std::mt19937 rand_src(12345);
  model::heisenberg1D h1(L,h,J);
  worm solver(beta, h1, 5);
  // std::vector<std::vector<int>> states;


  #if MESTIME
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  #endif


  int n_kink=0;
  for (int i=0; i < MCSTEP; i++){
    solver.init_states();
    solver.ops_sub.resize(0);
    for (int i=0; i< SWEEP; i++){
      solver.diagonal_update();
      solver.check_operators(solver.state, solver.ops_sub);
      solver.check_operators(solver.state, solver.ops_main);
      solver.worm_update();
      solver.swap_oplist();
    }
    n_kink += solver.ops_sub.size();
  }

  double energy = -((double)n_kink/MCSTEP) / beta;
  energy += h1.shifts[0] * h1.Nb;

  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "exection time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  #endif

  std::cout << "calculated energy : " << energy << std::endl;
  // std::cout << "argment : " << argv[1] << " is provided " << std::endl;
}