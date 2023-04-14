#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <random>
#include <math.h>
#include <bcl.hpp>
#include <assert.h> 
#include <fstream>
#include "outgoing_weight.hpp"
#include "funcs.hpp"
#include "operator.hpp"
#include "state_func.hpp"


namespace model{
using namespace std;
using VS = vector<size_t>;
using VVS = vector<VS>;
using VI = vector<int>;
using VVI = vector<VI>;
using VD = vector<double>;

template<class MC=bcl::heatbath> class local_operator;

// definition of equality operator.
template <class MC>
bool operator==(const local_operator<MC>& lhs, const local_operator<MC>& rhs)
{ 
  bool cmp = true;
  cmp &= (lhs._ham == rhs._ham);
  cmp &= lhs._ham_vector == rhs._ham_vector;
  cmp &= lhs._ham_prime == rhs._ham_prime;
  cmp &= lhs.leg == rhs.leg;
  cmp &= lhs.size == rhs.size;
  cmp &= lhs.ene_shift == rhs.ene_shift;
  cmp &= lhs.max_diagonal_weight_ == rhs.max_diagonal_weight_;
  cmp &= lhs.total_weights == rhs.total_weights;
  cmp &= lhs.signs == rhs.signs;
  cmp &= lhs.sps == rhs.sps;
  return cmp; 
}




/*
template argument
-------
MC : type of algorithm for generating transition matrix


params
-------
leg : number of sites bond operato acts on. typically 2.
size : number of hilbert space of bond operator.
sps : spin freedom per site.


variables
-------
TPROB : type of transition matrix
*/
typedef std::mt19937 engine_type;
typedef bcl::markov<engine_type> markov_t;

struct markov_v{
private:
  std::vector<markov_t> markov;
  std::vector<long long> s2i; 
public:
  markov_v(){}
  markov_v(std::vector<markov_t> markov, std::vector<long long> s2i)
  :markov(markov), s2i(s2i){}
  markov_t & operator[](size_t s) { 
    auto i = s2i[s];
    // auto i = s;
    if (i < 0) throw std::invalid_argument("probability that state s appears have to be 0"); 
    return markov[i]; 
  }
};

template <class MC>
class local_operator{
private:
  std::vector<double> _ham_vector;
  std::vector<bool> _has_warp; //check if the state has warphole
  std::vector<std::vector<double>> _ham_prime;
public:
  typedef MC MCT;
  double ham_vector(int i) {return _ham_vector[i];}
  const std::vector<double> & ham_vector() const {return _ham_vector;}
  const std::vector<std::vector<double>> & ham() const {return _ham;}
  const std::vector<std::vector<double>> & ham_prime() const {return _ham_prime;}

  std::vector<std::vector<double>> & single_flip(bool start, int spin) {
    //* i represents if the target site is start or the end of bond
    //* spin represents the spin of the site
    return _single_flip[spin + start*sps];
    }

  const double & single_flip(bool start, int spin, int x, int y) const {
    return _single_flip[spin + start*sps][x][y];
  }
  const bool has_warp(int i) const {return _has_warp[i];}
  using VECD = std::vector<double>;
  using TPROB = std::vector<VECD>; //type for transition probability. typically, this is 2D matrix with 4 x 4 elements( check notebook for detail definition of this type).
  std::vector<std::vector<std::vector<double>>> _single_flip;
  std::vector<std::vector<double>> _ham; // virtual hamiltonian (or maybe absolute of original hamiltonian)
  std::vector<int> signs; //list of sign defined via the sign of ham_prime;
  
  std::vector<TPROB> trans_prob; //num_configuration x 4 x 4 matrix.
  std::array<int, 2> num2index(int num);
  markov_v markov;
  std::vector<size_t> sps_base;
  outgoing_weight ogwt;

  const size_t sps;
  const int leg; // leg size.
  const int size; // size of operator (2**leg)
  double ene_shift = 0; //energy shift to ensure that diagonal elements of hamiltonian are non-negative
  double max_diagonal_weight_;
  double total_weights; //sum of diagonal elemtns of _ham



  local_operator(int leg, size_t sps = 2);

  void set_ham(double off_set = 0, double thres = 1E-6, bool dw = false, double alpha = 1 / 6.0);
  void set_trans_weights();
  void check_trans_prob();
  int index2num(std::array<int, 2> index);
  friend bool operator==<>(const local_operator<MC>& lhs, const local_operator<MC>& rhs);
};
  // { 
  //   bool cmp = true;
  //   cmp &= (lhs._ham == rhs._ham);
  //   cmp &= lhs._ham_vector == rhs._ham_vector;
  //   cmp &= lhs.ham_prime == rhs.ham_prime;
  //   cmp &= lhs.leg == rhs.leg;
  //   cmp &= lhs.size == rhs.size;
  //   cmp &= lhs.ene_shift == rhs.ene_shift;
  //   cmp &= lhs.max_diagonal_weight_ == rhs.max_diagonal_weight_;
  //   cmp &= lhs.total_weights == rhs.total_weights;
  //   cmp &= lhs.signs == rhs.signs;
  //   cmp &= lhs.sps == rhs.sps;
  //   return cmp; 
  // }

// template <class M>
// bool operator==(const local_operator<M>& lhs, const local_operator<M>& rhs) 
// { 
//   bool cmp = true;
//   cmp &= (lhs._ham == rhs._ham);
//   cmp &= lhs._ham_vector == rhs._ham_vector;
//   cmp &= lhs.ham_prime == rhs.ham_prime;
//   cmp &= lhs.leg == rhs.leg;
//   cmp &= lhs.size == rhs.size;
//   cmp &= lhs.ene_shift == rhs.ene_shift;
//   cmp &= lhs.max_diagonal_weight_ == rhs.max_diagonal_weight_;
//   cmp &= lhs.total_weights == rhs.total_weights;
//   cmp &= lhs.signs == rhs.signs;
//   cmp &= lhs.sps == rhs.sps;


//   return cmp; 
// }


template <class MC>
local_operator<MC>::local_operator(int leg, size_t sps)
  :leg(leg), size(pow(sps, leg)), ogwt(leg, sps), sps(sps)
  {
    _ham = std::vector<std::vector<double>>(size, std::vector<double>(size, 0));
    _single_flip.resize(2 * sps);
    for (int i = 0; i < 2 * sps; ++i){
      _single_flip[i] = std::vector<std::vector<double>>(sps, std::vector<double>(sps, std::numeric_limits<double>::max()));
    }
  }

/*
setting various variable for local_operators 
this function should be called after manually define local hamiltonian.

*params
------
boolean warp : 1 = have a chance for worm to warp.
double alpha : parameter which determines how much the diagonal elements of single flip will be allocated.
alpha may be 1 / (1 + Nb) where Nb is the number of NN bonds.
*/
template <class MC>
void local_operator<MC>::set_ham(double off_set, double thres, bool warp, double alpha){

  //! warp is not supported this version
  if (warp) throw std::invalid_argument("warp is not supported this version");

  if (0 > alpha || alpha >= 1) throw std::invalid_argument("alpha should be between 0 and 1");
  spin_state::StateFunc state_func(sps, 2);
  int N = size*size;
  ene_shift=0;
  _ham_prime = _ham;
  _ham_vector = std::vector<double>(N, 0);

  for (int i=0; i<_ham_prime.size();i++){
    ene_shift = std::min(ene_shift, _ham[i][i]);
  }
  ene_shift *= -1;
  ene_shift += off_set;
  for (int i=0; i<_ham_prime.size();i++){
    _ham_prime[i][i] += ene_shift;
  }

  //d* set single_flip operator and bond operator
  max_diagonal_weight_ = 0;
  for (int i=0; i<N; i++){
    auto index = state_func.num2state(i, 4);
    auto mat_index = num2index(i);
    _ham_vector[i] = _ham_prime[mat_index[0]][mat_index[1]];
    if (index[1] == index[3]) {
      if (index[0] == index[2]){
        single_flip(0, index[1])[index[0]][index[0]] = _ham_vector[i] / 2 * alpha;
      } else {
        single_flip(0, index[1])[index[0]][index[2]] = _ham_vector[i];
      }
    }
    // if (index[0] == index[2]) single_flip(1, index[0])[index[1]][index[3]] = _ham_vector[i];
    if (index[0] == index[2]){
      if (index[1] == index[3]){
        single_flip(1, index[0])[index[1]][index[1]] = _ham_vector[i] / 2 * alpha;
      } else {
        single_flip(1, index[0])[index[1]][index[3]] = _ham_vector[i];
      }
    }
    
    if (mat_index[0] == mat_index[1]) {
      _ham_vector[i] *= (1 - alpha);
      max_diagonal_weight_ = std::max(max_diagonal_weight_, _ham_vector[i]);
    } else if (index[0] == index[2] || index[1] == index[3])
    {
      _ham_vector[i] = 0;
    }
  }

  //d* check single_flip operator is real symmetric
  for (int s=0; s<sps; s++){
    for (int i=0; i<2; i++){
      for (int j=0; j<sps; j++){
        for (int k=0; k<sps; k++){
          if (std::abs(single_flip(i, s)[j][k] - single_flip(i, s)[k][j]) > 1e-8) {
            std::cerr << "difference is " << single_flip(i, s)[j][k] - single_flip(i, s)[k][j] << std::endl;
            throw std::runtime_error("single_flip is not symmetric");
            }
        }
      }
    }
  }

  total_weights = 0;
  for (int i=0; i<size; i++) {
  }

  for (int i=0; i<_ham_vector.size(); i++){
    auto& x = _ham_vector[i];
    signs.push_back(x >= 0 ? 1 : -1);
    x = std::abs(x);
    if (x < thres) x = 0;
  }

  std::vector<markov_t> markov_tmp;
  std::vector<long long> state2index;



  _has_warp.resize(_ham_vector.size(), false);
  state2index.resize(_ham_vector.size(), -1);
  if (leg > 2 && _ham_vector.size() > 1E5){
    std::cout << "too big size of _ham_vector for non-bond operator : In the case, only virtually one site operator at center is accepted" << std::endl;
    if (leg != 3) throw std::invalid_argument("leg more than 3 is not implemented yet");
    std::cerr << "sparse matrix may have a bug" << std::endl;
    exit(1);
    spin_state::Operator state(nullptr, &ogwt.pows, 0, 0, 0);
    for (size_t s=0; s < _ham_vector.size(); s++){
      state.set_state(s);
      
      if (state.get_local_state(0) != state.get_local_state(3) && state.get_local_state(2) != state.get_local_state(5)) continue;

      state2index[s] = markov_tmp.size();
      std::vector<double> ogw = ogwt.init_table(_ham_vector, s, warp);
      markov_tmp.push_back(markov_t(MC(), ogw));
    }
     markov = markov_v(markov_tmp, state2index);
  }else{
    for (size_t s=0; s < _ham_vector.size(); s++){
      state2index[s] = markov_tmp.size();
      std::vector<double> ogw = ogwt.init_table(_ham_vector, s, warp);
      if (warp) {
        for (int x=1; x<ogw.size(); x++) {_has_warp[s] = _has_warp[s] || ((ogw[x] != 0) & (ogw[0] != 0));}
      }
      markov_tmp.push_back(markov_t(MC(), ogw));
    }
    markov = markov_v(markov_tmp, state2index);
  }
}





template <class MC>
std::array<int, 2> local_operator<MC>::num2index(int num){
  ASSERT(num < size*size, "num is invalid");
  std::array<int, 2> index;
  index[0] = num%size;
  index[1] = num/size;
  return index;
}

template <class MC>
int local_operator<MC>::index2num(std::array<int, 2> index){
  ASSERT(index[0] < size && index[1] < size, "index is invalid");
  int num = 0;
  num += index[0];
  num += index[1] * size;
  return num;
}
}