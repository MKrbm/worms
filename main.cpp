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


  options opt(argc, argv, 16, 1, 1.0, "heisernberg");
  if (!opt.valid) std::exit(-1);
  int L = opt.L;
  int dim = opt.dim;
  double J = 1;
  double h = opt.H;
  double J1 = opt.J1;
  double J2 = opt.J2;
  std::string model_name = opt.MN;





  if (model_name == "heisernberg"){
    model::heisenberg spin_model(L,h,dim);
    exe_worm(spin_model, &opt);
  }else if (model_name == "shastry"){
    model::Shastry spin_model(L, J1, J2);
    exe_worm(spin_model, &opt);
  }else if (model_name == "shastry_v2"){
    model::Shastry_2 spin_model(L, J1, J2);
    exe_worm(spin_model, &opt);
  }else if (model_name == "test1"){
    model::test spin_model(L);
    exe_worm(spin_model, &opt);
  }else{
    std::cout << model_name << " is not avilable yet" << std::endl;
  }

}