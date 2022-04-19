/*
Majumdar-Ghosh model.
*/


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


  options* opt_ptr;

  if (argc > 2) {
    std::cout << "read from args " << std::endl;  
    options * opt;
    opt = new options(argc, argv, 16, 1, 1.0, "MG");
    opt_ptr = opt;
  }else{
    std::cout << "read txt file " << std::endl;  
    readConfig* config;
    config = new readConfig("../config/MG.txt", 2, 1, 1.0, "MG");
    opt_ptr = (options*)config;
    cout << "end read txt file" << endl;
  }


  if (!opt_ptr->valid) std::exit(-1);
  int L = opt_ptr->L;
  double sft = opt_ptr->shift;
  auto path_list = opt_ptr->path_list;
  auto path_list2 = opt_ptr->path_list;

  std::string model_name = opt_ptr->MN;
  std::cout << "model name is : " << model_name << std::endl;

  if (path_list.size()!= 1 && model_name == "MG") path_list = std::vector<std::string>({
    "../python/array/SS_ori_bond.npy",
  });

  typedef bcl::st2013 bcl_t;

  if (model_name == "MG"){
    model::MG<bcl_t> spin_model(path_list, L, opt_ptr->n_path, sft, opt_ptr->pom);
    exe_worm(spin_model, opt_ptr);
  }else if (model_name == "MG_2"){
    model::MG_2<bcl_t> spin_model(path_list, L, opt_ptr->n_path, sft, opt_ptr->pom);
    exe_worm(spin_model, opt_ptr);
  }
  delete opt_ptr;
}