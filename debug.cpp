// #define RANDOM_SEED 0
#include "MainConfig.h"
#include "options.hpp"

#include <iostream>
#include <worm.hpp>
#include <heisenberg.hpp>
#include <Shastry.hpp>
#include <string>
#include <chrono>
#include <observable.hpp>
#include <lattice/graph.hpp>
#include <lattice/coloring.hpp>

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


  options opt(argc, argv, 16, 1, 1.0, "heisernberg");
  if (!opt.valid) std::exit(-1);
  double beta = 1 / opt.T;
  int L = opt.L;
  int dim = opt.dim;
  double J = 1;
  double h = opt.H;
  std::string model_name = opt.MN;


  std::cout << "MC step : " << opt.sweeps << "\n" 
            << "thermal size : " << opt.therm << std::endl;

  BC::observable ene; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  BC::observable umag; // uniform magnetization 
  BC::observable ave_sign; // average sign 

  // std::cout << "size of int : " << sizeof(int8_t) << endl;


  // std::mt19937 rand_src(12345);
  // spin_model.lattice.print(std::cout);
  // if (model_name == "heisernberg"){
  // }else if (model_name == "shastry"){
  // }
  //* choose model 
  // model::heisenberg spin_model(L,h,dim);
  double J1 = 1;
  double J2 = 1;
  model::Shastry spin_model(L, J1, J2);

  worm<decltype(spin_model)> solver(beta, spin_model); //template needs for std=14
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
  // worm statistics
  double wcount = 0;
  double wlength = 0;
  double wdensity = spin_model.lattice.num_bonds();
  for (int i=0; i < opt.therm + opt.sweeps; i++){
    // solver.diagonal_update(); 
    solver.diagonal_update(wdensity); //n* need to be comment out 
    solver.worm_update(wcount, wlength);
    if (cnt >= opt.therm){
      int sign = 1;
      double mu = 0;
      for (const auto&  s : solver.state) {
        mu += 0.5 - s;
      }
      for (const auto& op : solver.ops_main){
        sign *= spin_model.loperators[op.op_type()].signs[op.state()];
        // if (sign == -1) {
        //   int x = 1;
        // }
      }
      double ene_tmp = - (double)solver.ops_main.size() / beta;
      for (int e=0; e<spin_model.Nop; e++){
        ene_tmp += spin_model.shifts[e] * spin_model.bond_t_size[e];
      }
      ene << ene_tmp * sign;
      // ene << (- ((double)solver.ops_main.size()) / beta + spin_model.shifts[0] * spin_model.lattice.num_bonds()) * sign;
      ave_sign << sign;
      mu /= spin_model.L;
      umag << mu * sign;
    }
    if (i <= opt.therm / 2) {
      if (wcount > 0) wdensity = spin_model.lattice.num_bonds()/ (wlength / wcount);
      if (i % (opt.therm / 8 + 1) == 0) {
        wcount /= 2;
        wlength /= 2;
      }
    }
    if (i == opt.therm / 2)
    std::cout << "Info: average number worms per MCS is reset from " << spin_model.L
              << " to " << wdensity << "\n\n";
    cnt++;
  }


  

  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
  #endif
  std::cout << "Elapsed time         = " << elapsed << " sec\n"
            << "Speed                = " << (opt.therm+opt.sweeps) / elapsed << " MCS/sec\n";
  std::cout << "Energy               = "
            << ene.mean()/ave_sign.mean() / spin_model.lattice.num_sites() << " +- " 
            << std::sqrt(std::pow(ene.error()/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2)) / spin_model.lattice.num_sites()
            << std::endl
            << "Uniform Magnetization  = "
            << umag.mean()/ave_sign.mean() << " +- " 
            << std::sqrt(std::pow(umag.error()/ave_sign.mean(), 2) + std::pow(umag.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2))
            << std::endl
            << "average sign           = "
            << ave_sign.mean() << " +- " << ave_sign.error() << std::endl;
}