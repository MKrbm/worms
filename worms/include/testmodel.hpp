#pragma once
#include "model.hpp"

namespace model{

  class test :public base_spin_model<1, 2>{
public:
    test(int L);
  };
}

