#ifndef __model__
#define __model__
#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <math.h>

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

  class local_operator;
  
  typedef std::vector<int> STATE;

  template <int N_op>
  class base_model_spin_1D;

  class heisenberg1D;



  /*
  sample from given discrete probability distribution
  prob : prob dist. the sum of all elements must be equal to 0
  r : random number (0 to 1)
  */
  template <typename PROB>
  int chooseAtRand (const PROB& prob, double r){
    double sum = 0;
    ASSERT(std::abs(std::accumulate(prob.begin(), prob.end(), 0)-1) <= TOR,
           "given prob can not be considered as probability distribution");
    int i;
    for(i=0; i<prob.size()-1; i++){
        sum += prob[i];
        if (sum >= r) break;
    }
    return i;
  }

}



class model::local_operator{
public:
  using VECD = std::vector<double>;
  using TPROB = std::vector<VECD>; //type for transition probability. typically, this is 2D matrix with 4 x 4 elements( check notebook for detail definition of this type).


  int L; // number of site operator acts 
  int size; // size of operator (2**L)
  std::vector<std::vector<double>> ham;
  std::vector<double> ham_vector;
  std::vector<std::vector<double>> trans_weights;
  std::vector<TPROB> trans_prob; //num_configuration x 4 x 4 matrix.
  std::vector<double> diagonal_cum_weight; //normalized diagonal elements;
  double total_weights; //sum of diagonal elemtns of ham
  std::array<int, 2> num2index(int num);


  local_operator(int L);
  local_operator();

  void set_ham();
  void set_trans_weights();
  void set_trans_prob();
  void check_trans_status(VECD, TPROB);
  void check_trans_prob();
  int index2num(std::array<int, 2> index);

  /*
  choose next dot from given random number
  params
  ------
  double r : random number
  int lnum : initial local state represent in integer (2 ** (2 * L) configurations)
  int cw_pos : current worm position (0 to 3)

  return
  ------
  int : pos worm goes out.
  */
  int choose_next_worm (double r,int lnum, int cw_pos) const{
    std::vector<double> const& trans = trans_prob[lnum][cw_pos];
    return chooseAtRand(trans, r);
  }


  void print_trans_weights(){
    for (int row=0; row<trans_weights.size(); row++){
        for(int column=0; column<trans_weights[0].size(); column++){
          printf("%.2f   ", trans_weights[row][column]);}
        printf("\n");
      }
  }
};

//$\hat{H} = \sum_{<i,j>} [J \vec{S}_i \dot \vec{S}_j - h/Nb (S_i^z + S_j^z)]$ 
// map spin to binary number e.g. -1 \rightarrow 0, 1 \rightarrow 1
template <int N_op>
// N_op : number of operators
class model::base_model_spin_1D{
public:
    const int L;
    const int Nb; // number of bonds.
    const bool PBC;
    static const int Nop = N_op; //number of local operator (1 for heisenberg model)
    double rho = 0;

    std::array<local_operator, N_op> loperators; //in case where there are three or more body interactions.
    std::array<int, N_op> leg_size; //size of local operators;
    std::array<double, N_op> operator_cum_weights;
    const std::vector<std::vector<int>> bonds;
    base_model_spin_1D(int L_, int Nb_, bool PBC, std::vector<std::vector<int>> bonds)
    :L(L_), Nb(Nb_), PBC(PBC), bonds(bonds){}
};



class model::heisenberg1D :public model::base_model_spin_1D<1>{
public:
    heisenberg1D(int L, double Jz, double Jxy, double h, bool PBC = true); //(1) 
    heisenberg1D(int L, double h, double Jz=1, bool PBC = true) : heisenberg1D(L, Jz, -Jz, h, PBC) {} //(2) : pass arguments to (1) constructor. This is for AFH.

    static std::vector<std::vector<int>> return_bonds(int L, bool PBC);
    double Jz, Jxy;
    const double h;

    int DopAtRand(double);
    void initial_setting();

};





#endif