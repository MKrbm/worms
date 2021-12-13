#include "../include/state.hpp"

/*
params
-----
int num : integer representation of state
int L : size of state

return
------
int[] state : binary representation (0 (spin up) and 1 (spin down)) of state.

*/
spin_state::STATE spin_state::num2state(int num, int L){
  int coef = 1;
  model::STATE state(L, 0); // all spin up
  for (int i=0; i<L; i++){
    state[i] = num&1;
    num /= 2;
  }
  return state;
}

std::string spin_state::return_name(int dot_type, int op_type){
  if (dot_type!=1) return dot_type_name[dot_type];
  else return dot_type_name[dot_type] + " ( " + op_type_name[op_type] + " )";
}