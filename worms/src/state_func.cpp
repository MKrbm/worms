#include "../include/state2.hpp"

namespace spin_state{

    StateFunc::StateFunc(size_t sps, size_t leg_size)
    :sps(sps), leg_size(leg_size), pows(pows_array(sps, leg_size))
    {}
    
    VS StateFunc::pows_array(size_t sps, size_t leg_size){
      VS arr(2*leg_size+1); size_t x = 1;
      for (int i=0; i<2*leg_size+1; i++) { arr[i]=x; x*=sps;}
      return arr;
    }

    size_t StateFunc::state2num(VUS const& state, int L){
      size_t num = 0;
      size_t x = 1;
      if (L < 0) L = state.size(); if (L == 0) return 0;
      for (int i = L-1; i >= 0; i--) { num += x*state[i]; x*=sps;}
      return num;
    }

    size_t StateFunc::state2num(VUS const& state, VS const& bond){
      size_t u = 0;
      for (int i=0; i<bond.size(); i++) u += (state[bond[i]] * pows[i]);
      return u;
  }
  
    VUS StateFunc::num2state(int num, int L){
      VUS state(L, 0); // all spin up
      for (int i=0; i<L; i++){ state[L-i-1] = num/pows[L-i-1]; num%=pows[L-i-1];}
      return state;
    }
}