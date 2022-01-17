#pragma once

#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
#include <vector>
#include "state.hpp"


namespace obs{
  using SPIN = model::SPIN;
  using size_t = std::size_t; 
  using STATE = model::STATE;
  using BOND = model::BOND;
  //* definition depends on number of spins per site (nls).

  template <size_t nls>
  int magnetization(STATE state){
    typedef spin_state::state_func<1> func;

    int mu;
    for (auto x : state) {
      auto local_state = func::num2state(x, nls);
      for (int y : local_state){
          mu += 0.5 - y;
      }
    }
    return mu;
  }
}