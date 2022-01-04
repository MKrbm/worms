#pragma once
#include "model.hpp"

namespace model{
  class heisenberg :public base_spin_model<1>{
public:
    heisenberg(int L, double Jz, double Jxy, double h, int dim); //(1) 
    heisenberg(int L, double h, int dim = 1, double Jz=1) : heisenberg(L, Jz, -Jz, h, dim) {} //(2) : pass arguments to (1) constructor. This is for AFH.

    double Jz, Jxy;
    const double h;
    int dim;
    void initial_setting();
};
}