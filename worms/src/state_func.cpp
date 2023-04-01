#include "../include/state_func.hpp"

namespace spin_state{

    StateFunc::StateFunc(size_t sps, size_t leg_size)
    :sps(sps), leg_size(leg_size)
    {}
    
    size_t StateFunc::state2num(VUS const& state, int L){
      size_t num = 0;
      size_t x = 0;
      if (L < 0) L = state.size(); if (L == 0) return 0;
      for (int i = L-1; i >= 0; i--) { x*=sps; num += state[i]; }
      return num;
    }

    size_t StateFunc::state2num(VUS const& state, VS const& bond){
      size_t u = 0;
      size_t L = bond.size();
      if (L != 2) throw std::invalid_argument("state2num: bond.size() != 2");

      for (int i = L-1; i >= 0; i--) {
        u *= sps;
        u += state[bond[i]];
        }
      return u;
  }
  
    VUS StateFunc::num2state(int num, int L){
      VUS state(L, 0); // all spin up
      for (int i = 0; i<L; i++){ 
        state[i] = num % sps; 
        num /= sps;
        }
      return state;
    }
}