
#include <uftree.hpp>
#include "../include/BC.hpp"
#include "../include/model.hpp"



// define functions for lolcal_operator class

model::local_operator::local_operator()
  :local_operator(2){}

model::local_operator::local_operator(int L)
  :L(L), size(pow(2, L)){
  ham = std::vector<std::vector<double>>(size, std::vector<double>(size, 0));
  ham_vector = std::vector<double>(size*size, 0);
  trans_weights = std::vector<std::vector<double>>(
          size*size,
          std::vector<double>(2*L)
          );
  diagonal_cum_weight = std::vector<double>(size, 0);
}


/*
setting various variable for local_operators 
this function should be called after manually define 2D local hamiltonian.

- set 1D hamiltonian 
*/
void model::local_operator::set_ham(){
  int N = ham_vector.size();

  for (int i=0; i<N; i++){
    auto index = num2index(i);
    ham_vector[i] = ham[index[0]][index[1]];
  }

  total_weights = 0;
  // for (int i=0; i<size; i++) total_weights+= ham[i][i];

  double tmp=0;
  for (int i=0; i<size; i++) {
    tmp += ham[i][i];
    diagonal_cum_weight[i] = tmp;
  }
  total_weights = *(diagonal_cum_weight.end()-1);

  set_trans_weights();//set trans_weights from ham_vector.
  set_trans_prob(); //set transition probability.

  check_trans_prob(); // check if transition probability is consistent with the definition of transition matrix

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

void model::local_operator::check_trans_status(VECD weights, TPROB transition_matrix){
    // check transition matrix
  std::cout << "[check transition matrix]\n";
  std::cout << "probability conservation = "
            << (BC::check_probability_conservation(transition_matrix) ? "pass" : "fail")
            << std::endl;
  std::cout << "detailed balance condition = "
            << (BC::check_detailed_balance(weights, transition_matrix) ? "pass" : "fail")
            << std::endl;
  std::cout << "balance condition = "
            << (BC::check_balance_condition(weights, transition_matrix) ? "pass" : "fail")
            << std::endl;
}


// check if transition probability is consistent with the definition of transition matrix
void model::local_operator::check_trans_prob(){
    int num_conf = trans_prob.size();
    ASSERT(num_conf == trans_weights.size(),
    "size of trans_prob and trans_weights must be the same");

    for(int i=0; i<num_conf; i++){
      ASSERT(BC::check(trans_weights[i], trans_prob[i])
      , "condition of transition matrix is not satisfied" );
    }
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


model::heisenberg1D::heisenberg1D(int L, double Jz, double Jxy, double h, bool PBC)
  :Jz(Jz), Jxy(Jxy),
  h(h), base_model_spin_1D(L, PBC ? L : L-1, PBC, return_bonds(L,PBC))
{
  std::cout << "model output" << std::endl;
  std::cout << "L : " << L << std::endl;
  std::cout << "Nb : " << Nb << std::endl;
  std::cout << "Jz : " << Jz << std::endl;
  std::cout << "h : " << h << std::endl;
  std::cout << "end \n" << std::endl;

  int l = 2;
  loperators[0] = local_operator(l);
  leg_size[0] = l;
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
  initial_setting(); //set_hum will be called inside the function


  // printf("\n\nprint trans weights : \n\n");
  // loperator.print_trans_weights();
  printf("end setting\n\n\n");

  rho = loperators[0].total_weights * Nb;
}

void model::heisenberg1D::initial_setting(){

  int i = 0;
  double tmp=0;
  for (auto& x : loperators){
    x.set_ham();
    tmp += x.total_weights;
    operator_cum_weights[i] = tmp;
  }
}







