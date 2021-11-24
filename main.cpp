#include <iostream>
#include <worm.hpp>
#include <model.hpp>
#include <string>
#include <chrono>

#include "MainConfig.h"
#define DEBUG 1
#define MCSTEP 20000
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


  heisenberg1D h1(L, J, true);
  solver loop(beta, h1);

  std::vector<std::vector<int>> states;


  #if MESTIME
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  #endif


  int n_kink=0;
  for (int i=0; i<MCSTEP; i++){
    for (int j=0; j<SWEEP; j++){
      loop.step();
    }
    n_kink += loop.get_kink_num();
  }

  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "exection time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  #endif

  double energy = -1.5 * ((double)n_kink/MCSTEP) / beta;
  std::cout << "calculated energy : " << energy << std::endl;
  // std::cout << "argment : " << argv[1] << " is provided " << std::endl;
}