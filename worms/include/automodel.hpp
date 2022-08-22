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
#include <fstream>
#include <tuple>
#include <fstream>
#include "localoperator.hpp"
#include "load_npy.hpp"
// #include "model.hpp"

namespace model{
  using namespace std;
  using VS = vector<size_t>;
  using VVS = vector<VS>;
  using VI = vector<int>;
  using VVI = vector<VI>;
  using VD = vector<double>;

  size_t num_type(std::vector<size_t> bond_type);

  class base_lattice;
  template <class MC=bcl::heatbath>
  class base_model;
}



class model::base_lattice{
public:

  const size_t L;
  const size_t Nb; // number of bonds.
  const size_t N_op;
  const VVS bonds;
  const VS bond_type;
  VS bond_t_size;
  // size_t max_l;
  double rho = 0;
  VD shifts;
  // VS sps_sites; 
  // VI leg_size; //size of local operators;
  // VS bond_t_size;

  base_lattice(int L, VVS bonds)
  :L(L), Nb(bonds.size()), N_op(1), bonds(bonds), bond_type(VS(bonds.size(), 0)){}

  base_lattice(int L, VVS bonds, VS bond_type)
  :L(L), Nb(bonds.size()), N_op(num_type(bond_type)),bonds(bonds), bond_type(bond_type){
    bond_t_size = VS(N_op, 0);
    for (int i=0; i<N_op; i++) for (auto bt : bond_type) if (bt==i) bond_t_size[i]++;
  }

  base_lattice(std::tuple<size_t, VVS, VS> tp)
  :base_lattice(get<0>(tp), get<1>(tp), get<2>(tp)){}

  base_lattice(std::string basis_name = "chain lattice", std::string cell_name = "simple1d", VS shapes = {6}, std::string file = "../config/lattice_xml.txt", bool print = false);


  //* initilizer function reading xml file.
  static std::tuple<size_t, VVS, VS> initilizer_xml(std::string basis_name, std::string cell_name, VS shapes, std::string file, bool print);
};

template <class MC>
class model::base_model : public model::base_lattice
{
public:
  using MCT = MC;
  const size_t dof; //degree of freedom
  const size_t leg_size = 2; //accepts only bond operators 
  const VS sps_sites; 
  VD shifts;
  std::vector<local_operator<MCT>> loperators;

  //* default constructor
  base_model( model::base_lattice lat, 
              int dof, 
              std::string ham_path, 
              VD params, 
              VI types, 
              double shift, 
              bool zero_worm, 
              bool repeat)
  :base_lattice(lat), dof(dof), sps_sites(VS(L, dof))
  {
    // cout << "hi" << endl;
    //* raed all numpy files in given path.
    std::vector<std::string> path_list;
    get_npy_path(ham_path, path_list);

    //* if repeat = true
    VI types_tmp;
    VD params_tmp;
    if (repeat){
      int r_cnt = N_op/types.size();
      if (r_cnt * types.size() != N_op) {std::cerr << "can not finish repeating types and params\n"; exit(1);}
      for (int i=0; i<r_cnt; i++) {
        types_tmp.insert(types_tmp.end(), types.begin(), types.end());
        params_tmp.insert(params_tmp.end(), params.begin(), params.end());
        for (auto &x : types) x += types.size();
      }
      cout << "repeat params " << r_cnt << " times." << endl;
      types = types_tmp;
      params = params_tmp;
    }

    //* check path_list
    if (path_list.size() != N_op){
      std::cerr << "# of operator does not match to # of bond types\n";
      exit(1);
    }

    //* check types
    if (params.size() != types.size()) {std::cerr << "size of params and types must match\n";exit(1);}
    VI _types(types);
    std::sort(_types.begin(), _types.end());
    int uniqueCount = std::unique(_types.begin(), _types.end()) - _types.begin();
    if ((size_t)N_op != 1+*std::max_element(_types.begin(), _types.end()) || N_op != uniqueCount)
    {  
      std::cerr << "types does not match requirements\n";
      exit(1);
    }

    //* load hamiltonians
    size_t op_label=0;
    for (int i=0; i<N_op; i++) loperators.push_back(local_operator<MC>(leg_size, dof)); 

    for (int l=0; l<path_list.size(); l++) {
      std::string path = path_list[l];
      auto pair = load_npy(path);
      VS shape = pair.first;
      VD data = pair.second;
      if (shape[0]!= shape[1]){ std::cerr << "require square matrix" << std::endl; exit(1); }
      size_t L = shape[0];
      if (L != pow(dof, leg_size)) {
        std::cerr << "dimenstion of given matrix does not match to dof ** legsize" << std::endl;
        std::cerr << "matrix size : " << L << std::endl; 
        exit(1); }

      std::cout << "hamiltonian is read from " << path << std::endl;
      local_operator<MCT>& loperator = loperators[types[op_label]];
      for (int i=0; i<shape[0]; i++) for (int j=0; j<shape[1]; j++)
      {
        auto x = data[i * shape[1] + j] * params[l];
        loperator.ham_rate[j][i] += x;
        loperator.ham[j][i] += x;
      }
      op_label++;
    }

    //* initial settings for local bond operators
    VD off_sets(N_op, shift);
    initial_setting(off_sets, 1E-8, zero_worm);
  }




  /*
  * initial setting function
  params
  ------
  off_sets : list of base shift for hamiltonians.
  boolean zw : 1 = zero worm. 
  thres : value lower than thres reduce to 0. usually 1E-8;
  */
  void initial_setting(VD off_sets, double thres, bool zw){
    int i = 0;
    double tmp=0;
    for (local_operator<MCT> & h : loperators){
      h.set_ham(off_sets[i], thres, zw);
      shifts.push_back(h.ene_shift);
      i++;
    }
  }
};


extern template class model::base_model<bcl::heatbath>;