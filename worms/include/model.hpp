#ifndef __model__
#define __model__
#include <iostream>
#include <stdio.h>
#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <random>
#include <math.h>
#include <bcl.hpp>
#include <lattice/graph.hpp>
#include <lattice/coloring.hpp>
#include <algorithm>
#include <assert.h> 
#include "outgoing_weight.hpp"
#include "load_npy.hpp"


#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

#define TOR 1.0e-10


#ifdef TOR
  #define DGREATER(X1, X2) (X1 >= X2-TOR)
#else
  #define DGREATER(X1, X2) (x1 >= X2)
  #define TOR 0
#endif

namespace model {

  template <int N_op, size_t _max_sps = 2, size_t _max_L = 4, class MC = bcl::heatbath>
  class base_spin_model;

  template <class MC = bcl::heatbath>
  class local_operator;
  
  using SPIN = unsigned short;
  using STATE = std::vector<SPIN>;
  using BOND = std::vector<std::size_t>;
  inline std::vector<BOND> generate_bonds(lattice::graph lattice){
    std::vector<BOND> bonds;
    for (int b=0; b<lattice.num_bonds(); b++){
      std::vector<size_t> tmp(2);
      tmp[0] = lattice.source(b);
      tmp[1] = lattice.target(b);
      bonds.push_back(tmp);
    }
    return bonds;
  }

  inline std::vector<size_t> generate_bond_type(lattice::graph lattice){
    std::vector<size_t> bond_type;
    for (int b=0; b<lattice.num_bonds(); b++) bond_type.push_back(lattice.bond_type(b));
    return bond_type;
  }

  inline size_t num_type(std::vector<size_t> bond_type){
    std::sort(bond_type.begin(), bond_type.end());
    auto it = std::unique(bond_type.begin(), bond_type.end());
    return std::distance(bond_type.begin(), it);
  }

  template <int N_op, size_t max_sps, size_t max_L, class MC>
  void set_hamiltonian(
    std::array<local_operator<MC>, N_op>& loperators, 
    std::array<int, N_op>& leg_size,
    std::vector<std::string> path_list, 
    std::vector<size_t> type_list,
    std::vector<double> coupling_list, 
    std::vector<std::string> path_list2 = std::vector<std::string>()
    ){
    ASSERT(path_list.size() == type_list.size(), "");
    ASSERT(path_list.size() == coupling_list.size(), "");

    if (path_list2.size() == 0) path_list2 = path_list;
    ASSERT(path_list2.size() == path_list.size(), "");
    ASSERT(leg_size.size() == N_op, "");
    ASSERT((size_t)N_op == 1+*std::max_element(type_list.begin(), type_list.end()), " ");

    for (int l=0; l<N_op; l++)
    {  
      loperators[l] = local_operator<MC>(leg_size[l], max_sps);
      ASSERT(loperators[l].size == pow(max_sps, leg_size[l]),"size is inconsistent, the size of hamiltonian should be fixed to max_sps ** leg");
      for (int i=0; i<loperators[l].size; i++)
        for (int j=0; j<loperators[l].size; j++) { loperators[l].ham[j][i] = 0; loperators[l].ham_rate[j][i] = 0;}
    }

    int op_label = 0;


    for (int i=0; i<path_list.size(); i++) {
      auto path = path_list[i];
      auto pair = load_npy(path);
      auto shape = pair.first;
      auto data = pair.second;

      auto path2 = path_list2[i];
      auto pair2 = load_npy(path2);
      auto shape2 = pair2.first;
      auto data2 = pair2.second;

      if (shape[0]!=shape2[0] || shape[1]!=shape2[1]){
         std::cerr << "shape is inconsistent" << std::endl;
      }
      // int l = 2;
      size_t op_type = type_list[op_label];
      std::cout << "hamiltonian is read from " << path << std::endl;
      ASSERT(shape[0] == shape[1],"loaded local hamiltonian is not squared matrix");
      ASSERT(loperators[op_type].size == shape[0], "loaded local hamiltonian conflict with loperator in size");
      for (int i=0; i<shape[0]; i++){
        for (int j=0; j<shape[1]; j++)
        {
          auto x = coupling_list[op_label]*data[i * shape[1] + j];
          auto x2 = coupling_list[op_label]*data2[i * shape[1] + j];

          loperators[op_type].ham_rate[j][i] += x;
          loperators[op_type].ham[j][i] += x2;
        
        }
      }
      op_label++;
    }

  };
}


/*
*params
-------
leg : number of sites bond operato acts on. typically 2.
size : number of hilbert space of bond operator.
sps : spin freedom per site.

*template argument
-------
MC : type of algorithm for generating transition matrix

*variables
-------
TPROB : type of transition matrix
*/
template <class MC>
class model::local_operator{
public:
  using VECD = std::vector<double>;
  using TPROB = std::vector<VECD>; //type for transition probability. typically, this is 2D matrix with 4 x 4 elements( check notebook for detail definition of this type).
  typedef std::mt19937 engine_type;
  typedef bcl::markov<engine_type> markov_t;
  typedef MC MCT;
  outgoing_weight ogwt;

  size_t sps;
  int leg; // leg size.
  int size; // size of operator (2**leg)
  double ene_shift = 0; //energy shift to ensure that diagonal elements of hamiltonian are non-negative
  double max_diagonal_weight_;
  double total_weights; //sum of diagonal elemtns of ham

  std::vector<std::vector<double>> ham;
  std::vector<std::vector<double>> ham_;
  std::vector<double> ham_vector;
  std::vector<double> ham_rate_vector;
  std::vector<std::vector<double>> ham_rate;
  std::vector<int> signs; //list of sign defined via the sign of ham_;
  std::vector<TPROB> trans_prob; //num_configuration x 4 x 4 matrix.
  std::array<int, 2> num2index(int num);
  std::vector<markov_t> markov;
  std::vector<size_t> sps_base;


  local_operator(int leg, size_t sps = 2);
  local_operator();

  void set_ham(double off_set = 0, double thres = 1E-8, bool dw = false);
  void set_trans_weights();
  void check_trans_prob();
  int index2num(std::array<int, 2> index);
};

/*
//$\hat{H} = \sum_{<i,j>} [J \vec{S}_i \dot \vec{S}_j - h/Nb (S_i^z + S_j^z)]$ 
// map spin to binary number e.g. -1 \rightarrow 0, 1 \rightarrow 1
* S is local freedomness.

*params
------
sps : spin freedom per site.

*member variables
loperators : lists of local operator. the size of lists corresponds to N_op.

*template arguments
------
N_op : number of types of operatos. (heisernberg = 1, shastry = 2)
max_L : maximum leg size of bond operators. typically 4. 
MC : type of algorithm for generating transition matrix


*/
template <int N_op, size_t _max_sps, size_t _max_L, class MC>
class model::base_spin_model{
protected:

public:
  static const size_t max_L = _max_L;
  static const int Nop = N_op;
  static const size_t max_sps = _max_sps;
  static const size_t max_sps2 = _max_sps;
  typedef MC MCT;

  const int L;
  const int Nb; // number of bonds.
  const std::vector<BOND> bonds;
  const std::vector<size_t> bond_type;
  std::vector<size_t> sps_sites; 

  double rho = 0;

  std::array<local_operator<MCT>, N_op> loperators;
  std::array<int, N_op> leg_size; //size of local operators;
  std::array<size_t, N_op> bond_t_size;
  std::vector<double> shifts;
  lattice::graph lattice;

  base_spin_model(int L_, int Nb_, std::vector<BOND> bonds)
  :L(L_), Nb(Nb_), bonds(bonds){}

  base_spin_model(lattice::graph lt, std::vector<size_t> sps_list)
  :base_spin_model(lt)
  {
    ASSERT(sps_list.size() == L, "size of sps_list is inconsistent with model size L " );
    sps_sites = sps_list;
  }


  base_spin_model(lattice::graph lt)
  :L(lt.num_sites()), Nb(lt.num_bonds()), lattice(lt), 
    bonds(generate_bonds(lt)), bond_type(generate_bond_type(lt))
  {
    int sum = 0;
    for (int i=0; i<Nop; i++){
      bond_t_size[i] = 0;
      for (auto bt : bond_type){
        if (bt==i) bond_t_size[i]++;
      }
      sum += bond_t_size[i];
    }

    if (num_type(bond_type)!=Nop) {
      std::cerr << "Nop is not consistent with number of bond_type" << std::endl;
      std::terminate();
    }

    if (sum != bonds.size()) {
      std::cerr << "something wrong in bond_type" << std::endl;
      std::terminate();
    }

    sps_sites = std::vector<size_t>(L, _max_sps);
  }

  /*
  *params
  ------
  boolean dw : 1 = have a chance to delete a worm while updating.
  */

  void initial_setting(std::vector<double>off_sets = std::vector<double>(N_op,0), double thres = 1E-8, bool dw = false){
    int i = 0;
    double tmp=0;
    for (auto& x : loperators){
      x.set_ham(off_sets[i], thres, dw);
      shifts.push_back(x.ene_shift);
      i++;
    }
  }

};





// define functions for lolcal_operator class
template <class MC>
model::local_operator<MC>::local_operator()
  :local_operator(2){}

template <class MC>
model::local_operator<MC>::local_operator(int leg, size_t sps)
  :leg(leg), size(pow(sps, leg)), ogwt(leg, sps), sps(sps){

  if (sps<=0) size = (1<<leg); // default size is 2**leg.
  ham = std::vector<std::vector<double>>(size, std::vector<double>(size, 0));
  ham_rate = std::vector<std::vector<double>>(size, std::vector<double>(size, 0));
  ham_vector = std::vector<double>(size*size, 0);
  ham_rate_vector = std::vector<double>(size*size, 0);

}


/*
setting various variable for local_operators 
this function should be called after manually define 2D local hamiltonian.

- set 1D hamiltonian 

*params
------
boolean dw : 1 = have a chance to delete a worm while updating.
*/
template <class MC>
void model::local_operator<MC>::set_ham(double off_set, double thres, bool dw){
  int N = ham_vector.size();
  ene_shift=0;
  ham_ = ham;

  for (int i=0; i<ham_.size();i++){
    ene_shift = std::min(ene_shift, ham[i][i]);
    ene_shift = std::min(ene_shift, ham_rate[i][i]);
  }
  ene_shift *= -1;
  ene_shift += off_set;
  for (int i=0; i<ham_.size();i++){
    ham_[i][i] += ene_shift;
    ham_rate[i][i] += ene_shift;
  }

  for (int i=0; i<N; i++){
    auto index = num2index(i);
    ham_vector[i] = ham_[index[0]][index[1]];
    ham_rate_vector[i] = ham_rate[index[0]][index[1]];
    if (std::abs(ham_vector[i]) < thres) ham_vector[i] = 0;
    if (std::abs(ham_rate_vector[i]) < thres) ham_rate_vector[i] = 0;
  }


  total_weights = 0;
  // for (int i=0; i<size; i++) total_weights+= ham[i][i];

  double tmp=0;
  max_diagonal_weight_ = 0;
  for (int i=0; i<size; i++) {
    tmp += ham_[i][i];
    max_diagonal_weight_ = std::max(max_diagonal_weight_, ham_[i][i]);
  }



  // max_diagonal_weight_ = std::max(max_diagonal_weight_, weights_[p]);

  for (int i=0; i<ham_vector.size(); i++){
    auto& x = ham_vector[i];
    auto& y = ham_rate_vector[i];
    signs.push_back(x >= 0 ? 1 : -1);
    x = std::abs(x);

    if (y!=0 && x == 0){
      std::cerr << "cannot reweighting since support doesn't cover the original matrix" << std::endl;
    }
    if (x!= 0) y = y/x;
  }

  // set transition probability
  ogwt.init_table(ham_vector, dw);
  for (int c = 0; c < ogwt.size(); ++c) markov.push_back(markov_t(MC(),ogwt[c]));

  // auto rand_src = engine_type(2021);
  // auto xxx = markov[0](0, rand_src);



  // check_trans_prob(); // check if transition probability is consistent with the definition of transition matrix

}




template <class MC>
std::array<int, 2> model::local_operator<MC>::num2index(int num){
  ASSERT(num < size*size, "num is invalid");
  std::array<int, 2> index;
  index[0] = num%size;
  index[1] = num/size;
  return index;
}

template <class MC>
int model::local_operator<MC>::index2num(std::array<int, 2> index){
  ASSERT(index[0] < size && index[1] < size, "index is invalid");
  int num = 0;
  num += index[0];
  num += index[1] * size;
  return num;
}




#endif