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
#include <algorithm>
#include <assert.h> 
#include <iostream>
#include <string>
#include <fstream>

#include "model.hpp"

namespace model{
  template <size_t MAX_L = 4, class MC = bcl::heatbath>
  class base_model;
}


template <size_t MAX_L, class MC>
class model::base_model{
public:
  static const size_t max_L = MAX_L;
  typedef MC MCT;

  const int L;
  const int Nb; // number of bonds.
  const int N_op;
  const std::vector<BOND> bonds;
  const std::vector<size_t> bond_type;
  std::vector<size_t> sps_sites; 
  double rho = 0;
  std::vector<local_operator<MCT>> loperators;
  std::vector<int> leg_size; //size of local operators;
  std::vector<size_t> bond_t_size;
  std::vector<double> shifts;

  base_model(int L, std::vector<BOND> bonds)
  :L(L), Nb(bonds.size()), N_op(1), bonds(bonds), bond_type(std::vector<size_t>(bonds.size(), 0)){}

  base_model(int L, std::vector<BOND> bonds, std::vector<size_t> bond_type, std::vector<size_t> sps_sites)
  :L(L), bonds(bonds), bond_type(bond_type), sps_sites(sps_sites){}

  base_model(std::string file = "../config/lattice_xml.txt", std::string basis_name = "chain lattice", std::string cell_name = "simple1d");
};