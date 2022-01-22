// #define RANDOM_SEED 0
#include "exec.hpp"

// #define DEBUG 1
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


  // options opt(argc, argv, 16, 1, 1.0, "heisernberg");
  readConfig config("../config/shastry.txt", 2, 1, 1.0, "shastry_v2");

  if (!config.valid) std::exit(-1);
  int L = config.L;
  int dim = config.dim;
  double J = 1;
  double h = config.H;
  double J1 = config.J1;
  double J2 = config.J2;
  std::string model_name = config.MN;


  if (model_name != "shastry_v2"){
    model::Shastry spin_model(L, J1, J2);
    exe_worm(spin_model, config);
  }else{
    model::Shastry_2 spin_model(L, J1, J2);
    exe_worm(spin_model, config);
  }
}