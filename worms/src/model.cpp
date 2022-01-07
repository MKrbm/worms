
#include <uftree.hpp>
#include "../include/BC.hpp"
#include "../include/model.hpp"



// define functions for lolcal_operator class

model::local_operator::local_operator()
  :local_operator(2){}

model::local_operator::local_operator(int leg, size_t sps)
  :leg(leg), size(pow(sps, leg)), ogwt(leg), sps(sps){

  if (sps<=1) size = pow(2,leg); // default size is 2**leg.
  ham = std::vector<std::vector<double>>(size, std::vector<double>(size, 0));
  ham_vector = std::vector<double>(size*size, 0);
  diagonal_cum_weight = std::vector<double>(size, 0);
}


/*
setting various variable for local_operators 
this function should be called after manually define 2D local hamiltonian.

- set 1D hamiltonian 
*/
void model::local_operator::set_ham(double off_set){
  int N = ham_vector.size();
  ene_shift=0;
  ham_ = ham;

  for (int i=0; i<ham_.size();i++){
    ene_shift = std::min(ene_shift, ham[i][i]);
  }
  ene_shift *= -1;
  ene_shift += off_set;
  for (int i=0; i<ham_.size();i++){
    ham_[i][i] = ham_[i][i] + ene_shift;
  }

  for (int i=0; i<N; i++){
    auto index = num2index(i);
    ham_vector[i] = ham_[index[0]][index[1]];
  }


  total_weights = 0;
  // for (int i=0; i<size; i++) total_weights+= ham[i][i];

  double tmp=0;
  max_diagonal_weight_ = 0;
  for (int i=0; i<size; i++) {
    tmp += ham_[i][i];
    diagonal_cum_weight[i] = tmp;
    max_diagonal_weight_ = std::max(max_diagonal_weight_, ham_[i][i]);
  }



  // max_diagonal_weight_ = std::max(max_diagonal_weight_, weights_[p]);

  for (auto& x : ham_vector){
    signs.push_back(x >= 0 ? 1 : -1);
    x = std::abs(x);
  }
  total_weights = *(diagonal_cum_weight.end()-1);

  // set transition probability
  ogwt.init_table(ham_vector);
  for (int c = 0; c < ogwt.size(); ++c) markov.push_back(markov_t(bcl::st2010(), ogwt[c]));

  // auto rand_src = engine_type(2021);
  // auto xxx = markov[0](0, rand_src);



  // check_trans_prob(); // check if transition probability is consistent with the definition of transition matrix

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



// template<>
// model::base_spin_model<1>::base_spin_model(lattice::graph lt)
// :L(lt.num_sites()), Nb(lt.num_bonds()), lattice(lt),
//   bonds(generate_bonds(lt)), bond_type(generate_bond_type(lt))
// {
//   std::cout << "N_OP = 1" << std::endl;
// }

//end definition








