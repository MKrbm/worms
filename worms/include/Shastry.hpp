#pragma once
#include "model.hpp"

namespace model{
  class Shastry :public base_spin_model<1>{
public:
  Shastry(int L, double J1, double J2 = 1, double h = 0, int dim = 1); //(1) 
  const double J1,J2,h;
  int dim;
  void initial_setting();
};
}

