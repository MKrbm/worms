
#include <uftree.hpp>
#include "../include/BC.hpp"
#include "../include/model.hpp"



// define functions for lolcal_operator class

model::local_operator::local_operator(int L)
  :L(L), size(pow(2, L)){
  ham = std::vector<std::vector<double>>(size, std::vector<double>(size, 0));
  ham_vector = std::vector<double>(size*size, 0);
  trans_weights = std::vector<std::vector<double>>(
          size*size,
          std::vector<double>(2*L)
          );
}

void model::local_operator::set_ham(){
  int N = ham_vector.size();

  for (int i=0; i<N; i++){
    auto index = num2index(i);
    ham_vector[i] = ham[index[0]][index[1]];
  }
  set_trans_weights();//set trans_weights from ham_vector.
  set_trans_prob(); //set transition probability.
}


void model::local_operator::set_trans_prob(){
  trans_prob.resize(0);
  int N = trans_weights.size(); // sizes*size
  int M = trans_weights[0].size(); // 2 * L

  for(const auto& x : trans_weights){
    trans_prob.push_back(BC::metropolis(x));
  }
}


/*
set weights for transition
the first index represent the sate when the worm invade into the operator
e.g. 
                     ↑
                     * 
1 0        1 0     1 1
---        -*-     ---
1 0    →   0 0  →  0 0
↑
*

In this case, first index = [1, 0, 0, 0] (1*2^3 + 0 + 0 + 0 = 8)
and the second index represents which bond the worm choese;
In above case, j = 3.
*/
void model::local_operator::set_trans_weights(){
  // ham_vector.size() = size*size;
  for(int i=0; i<ham_vector.size(); i++)
    for (int j=0; j<2*L; j++){
      trans_weights[i][j] = ham_vector[i ^ (1<<j)];
    }
}



std::array<int, 2> model::local_operator::num2index(int num){
  ASSERT(num < size*size, "num is invalid");
  std::array<int, 2> index;
  index[0] = num%size;
  index[1] = num/size;
  return index;
}

int model::local_operator::index2num(std::array<int, 2> index){
  ASSERT(index[0] < size && index[1] < size, "index is invalid");
  int num = 0;
  num += index[0];
  num += index[1] * size;
  return num;
}

//end definition





std::vector<std::vector<int>> model::heisenberg1D::return_bonds(int L, bool PBC){
  std::vector<std::vector<int>> bonds(L, std::vector<int>(2));

  for(int i=0; i<L; i++){
    bonds[i][0] = i;
    bonds[i][1] = (i+1)%L;
  }
  // std::vector<std::vector<int>> vtr {{34,55},{45},{53,62}};
  return bonds;
}

const int model::heisenberg1D::N_op = 3;

model::heisenberg1D::heisenberg1D(int L, double Jz, double Jxy, double h, bool PBC)
  :L(L), PBC(PBC), Jz(Jz), Jxy(Jxy), bonds(return_bonds(L,PBC)),
  Nb(PBC ? L : L-1), h(h), base_model_spin_1D(L, PBC ? L : L-1, 3)
{
  std::cout << "model output" << std::endl;
  std::cout << "L : " << L << std::endl;
  std::cout << "Nb : " << Nb << std::endl;
  std::cout << "Jz : " << Jz << std::endl;
  std::cout << "h : " << h << std::endl;
  std::cout << "end \n" << std::endl;

  loperators.push_back(local_operator(2));
  auto& loperator = loperators[0];
  // set hamiltonian
  loperator.ham[0][0] = h;
  loperator.ham[1][1] = (h+1)/2.0;
  loperator.ham[2][2] = (h+1)/2.0;
  loperator.ham[1][2] = 1/2.0;
  loperator.ham[2][1] = 1/2.0;
  //end

  
  printf("set local hamiltonian : \n\n");
  for (int row=0; row<loperator.size; row++)
  {
    for(int column=0; column<loperator.size; column++)
    {
      printf("%.2f   ", loperator.ham[row][column]);}
   
    printf("\n");
  }
  loperator.set_ham();

  printf("\n\nprint trans weights : \n\n");
  loperator.print_trans_weights();
  printf("end setting\n\n\n");

  rho = h*Nb + (1+h)/2 * Nb;
}

/*
pick diagonal operator type at random for given r ~ uniform(0,1)
*/
int model::heisenberg1D::DopAtRand(double r){
  double sum = 0;
  int i;
  for(i=0; i<NDop-1; i++){
    sum += prob[i];
    if (sum >= r) break;
  }

  return i;
}


/*
params
-----
int[] state : vector of 1 or -1. 
int L : size of state

return
------
integer representation of state

*/
int model::state2num(model::STATE state, int L = -1){
  int num = 0;
  int coef = 1;
  if (L < 0) L = state.size();
  for (int i = 0; i < L; i++) {
    num += ((1-state[i])/2) * coef;
    coef *= 2;
  }
  return num;
}


/*
params
-----
int num : integer representation of state
int L : size of state

return
------
int[] state : binary representation (1 and -1 instead of 1 and 0) of state.

*/
model::STATE model::num2state(int num, int L){
  int coef = 1;
  model::STATE state(L, -1);
  for (int i=0; i<L; i++){
    state[i] = 1^(num&1);
    num /= 2;
  }
  return state;
}


