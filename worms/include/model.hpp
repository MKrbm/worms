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
#include <lattice/graph.hpp>
#include <lattice/coloring.hpp>
#include <algorithm>
#include <assert.h> 
#include "outgoing_weight.hpp"
#include "load_npy.hpp"
#include "localoperator.hpp"


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

  template <int N_op, size_t _max_sps = 2, size_t MAX_L = 4, class MC = bcl::heatbath>
  class base_spin_model;

  // template <class MC = bcl::heatbath>
  // class local_operator;
  
  using SPIN = unsigned short;
  using STATE = std::vector<SPIN>;
  using BOND = std::vector<std::size_t>;

  // in source file.
  std::vector<BOND> generate_bonds(lattice::graph lattice);
  std::vector<size_t> generate_bond_type(lattice::graph lattice);
  size_t num_type(std::vector<size_t> bond_type);
  /*
  params
  ------
  leg_size : list of legsize. legsize is number of leg for operator. 2 for bond opeerator
  
  type_list : list of types. type specify which local hamiltonian is applied to which bond type. 

  path_list : list of real local hamiltonian. ham_rate holds the rate local/virtual.
  path_list2 : list of virtual local hamiltonian. loperator.ham holds this value.
  */

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
      auto pair = load_npy<std::complex<double>>(path);
      auto shape = pair.first;
      auto data = pair.second;

      auto path2 = path_list2[i];
      auto pair2 = load_npy<std::complex<double>>(path2);
      auto shape2 = pair2.first;
      auto data2 = pair2.second;

      if (shape[0]!=shape2[0] || shape[1]!=shape2[1]){
         std::cerr << "shape is inconsistent" << std::endl;
      }
      // int l = 2;
      size_t op_type = type_list[op_label];
      std::cerr << "hamiltonian is read from " << path << std::endl;
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



template <int N_op, size_t _max_sps, size_t MAX_L, class MC>
class model::base_spin_model{
protected:

public:
  static const size_t max_L = MAX_L;
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
    using namespace std;
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
  boolean zw : 1 = zero worm.
  */

  void initial_setting(std::vector<double>off_sets = std::vector<double>(N_op,0), double thres = 1E-8, bool zw = false){
    int i = 0;
    double tmp=0;
    for (auto& x : loperators){
      x.set_ham(off_sets[i], thres, zw);
      shifts.push_back(x.ene_shift);
      i++;
    }
  }

};


