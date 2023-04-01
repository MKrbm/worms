#pragma once

#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
#include <vector>

using namespace std;
using VS = vector<size_t>;
using VVS = vector<VS>;
using VI = vector<int>;
using VVI = vector<VI>;
using VD = vector<double>;
using US = unsigned short;
using VUS = vector<US>;


namespace spin_state{
  struct StateFunc{
  public:
    const size_t sps;
    const size_t leg_size;
    StateFunc(size_t sps, size_t leg_size);
    size_t state2num(VUS const&, int = -1);
    size_t state2num(VUS const&, VS const&);
    VUS num2state(int, int);
  };
}
