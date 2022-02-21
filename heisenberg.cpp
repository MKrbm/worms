// #define RANDOM_SEED 0
#include "exec.hpp"
#include "heisenberg.hpp"

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
  // readConfig config("../config/heisernberg.txt", 16, 1, 1.0, "heisernberg");

  options* opt_ptr;

  if (argc > 2) {
    std::cout << "read from args " << std::endl;  
    options * opt;
    opt = new options(argc, argv, 16, 1, 1.0, "shastry");
    opt_ptr = opt;
  }else{
    std::cout << "read txt file " << std::endl;  
    readConfig* config;
    config = new readConfig("../config/heisernberg.txt", 2, 1, 1.0, "heisernberg");
    opt_ptr = (options*)config;
  }


  if (!opt_ptr->valid) std::exit(-1);
  int L = opt_ptr->L;
  int dim = opt_ptr->dim;
  double J = 1;
  double h = opt_ptr->H;
  double J1 = opt_ptr->J1;
  double J2 = opt_ptr->J2;
  double sft = opt_ptr->shift;
  auto path_list = opt_ptr->path_list;
  std::string model_name = opt_ptr->MN;
  std::cout << "model name is : " << model_name << std::endl;
  if (path_list.size()<3) path_list = std::vector<std::string>({
    "../python/array/H_bond_z.npy",
    "../python/array/H_bond_x.npy",
    "../python/array/H_bond_y.npy",
  });


  typedef bcl::st2013 bcl_t;
  // model::heisenberg<bcl_t> spin_model(L,h,dim);

  if (model_name != "heisernberg_v2"){
    // model::Shastry<bcl_t> spin_model(L, J1, J2);
    model::heisenberg<bcl_t> spin_model(L,J1, J2, h,dim);
    exe_worm(spin_model, opt_ptr);
  }else{
    model::heisenberg_v2<bcl_t> spin_model(path_list, L, J1, J2, h, dim, sft, opt_ptr->pom);
    exe_worm(spin_model, opt_ptr);
  }
}