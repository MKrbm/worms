#pragma once
#include "model.hpp"
#include "Shastry.hpp"

namespace model{
  class test :public base_spin_model<1, 2>{
    typedef base_spin_model<1, 2> MCT;
public:
    test(int L);
  };
}


model::test::test(int L)
:MCT(lattice::graph::simple(1, L))
{
  int local = 0;
  for (auto path : {"../python/array/test_model.npy"}) {
    auto pair = load_npy(path);
    auto shape = pair.first;
    auto data = pair.second;
    int l = 2; //* leg size
    loperators[local] = local_operator<>(l, base_spin_model::nls); 
    leg_size[local] = l;
    std::cout << "hamiltonian is read from " << path << std::endl;
    for (int i=0; i<shape[0]; i++){
      for (int j=0; j<shape[1]; j++)
      {
        auto x = data[i * shape[1] + j];
        if (std::abs(x) > 1E-4) {
          loperators[local].ham[j][i] = x;
          printf("[%2d, %2d] : %3.2f\n", j, i, x);
          }
      }
    }
    std::cout << "\n\n" << std::endl;
    local ++;
  }
  std::vector<double> off_sets(1,0.0);

  initial_setting(off_sets);  

  for (int i=0; i<shifts.size(); i++){
    printf("shifts[%d] = %f\n\n", i, shifts[i]);
  }
}

