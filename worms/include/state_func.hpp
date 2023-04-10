#pragma once

#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
#include <vector>
#include "spin_state.hpp"


namespace spin_state{
  struct StateFunc{
  public:
    const size_t sps;
    const size_t leg_size;
    StateFunc(size_t sps, size_t leg_size);
    size_t state2num(state_t const&, int = -1);
    size_t state2num(state_t const&, VS const&);
    state_t num2state(int, int);
  };
}
