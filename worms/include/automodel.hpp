#pragma once
#include <stdio.h>

#include <array>
#include <bcl.hpp>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "lattice/types.hpp"
#include "localoperator.hpp"
#include "model.hpp"

namespace model {
size_t num_type(std::vector<size_t> bond_type);

class base_lattice;
template <class MC = bcl::heatbath>
class base_model;

struct BondTargetType {
  size_t bt;
  bool start;     //* true if the target site is start of the bond.
  size_t target;  //* target site.
  BondTargetType(size_t bt, bool start, size_t target)
      : bt(bt), start(start), target(target) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const BondTargetType &bondTarget) {
    os << "{ bt: " << bondTarget.bt
       << ", start: " << (bondTarget.start ? "true" : "false")
       << ", target: " << bondTarget.target << " }";
    return os;
  }
};
}  // namespace model

class model::base_lattice {
 private:
  VS bond_t_legsize;

 protected:
  vector<VVS> type2bonds;
  VS bond_t_size;

 public:
  const size_t L;
  const size_t Nb;  // number of bonds.
  const size_t N_op;
  const VVS bonds;
  const VS bond_type;
  const VS site_type;
  vector<vector<BondTargetType>> nn_sites;

  double rho = 0;

  base_lattice(int L, VVS bonds);
  base_lattice(int L, VVS bonds, VS bond_type, VS site_type);
  base_lattice(std::tuple<size_t, VVS, VS, VS> tp);
  base_lattice(std::string basis_name = "chain lattice",
               std::string cell_name = "simple1d", VS shapes = {6},
               std::string file = "../config/lattice_xml.txt",
               bool print = false,
               lattice::boundary_t = lattice::boundary_t::periodic
               );
  static std::tuple<size_t, VVS, VS, VS> initilizer_xml(
      std::string basis_name, std::string cell_name, VS shapes,
      std::string file, bool print,
      lattice::boundary_t boundary);
};

/*
params
------

  lat : lattice

  dofs : degree of freedom

  ham_path : path of hamiltonian files

  types : there may possibility that adding different operators on the same
bond.

*/
template <class MC>
class model::base_model : public model::base_lattice {
 private:
  VD shifts;
  double origin_shift;

 public:
  VS _sps_sites;  // degree of freedom
  using MCT = MC;
  const size_t leg_size = 2;  // accepts only bond operators
  std::vector<local_operator<MCT>> loperators;
  std::vector<double> s_flip_max_weights;
  double alpha;
  bool zw;  // zero worm

  //* default constructor with no arguments
  /*
  constructor
  */
  base_model(model::base_lattice lat, VS dofs, std::string ham_path, VD params,
             VI types, double shift, bool zero_worm, bool repeat,
             bool print = true, double alpha = 0);

  base_model(model::base_lattice lat, VS dofs, std::string ham_path,
             std::string u_path, VD params, VI types, double shift,
             bool zero_worm, bool repeat, bool print = true, double alpha = 0);

  //* simple constructor
  base_model(model::base_lattice lat, VS dofs,
             std::vector<std::vector<std::vector<double>>> hams, double shift,
             bool zero_worm);

  /*
  * initial setting function
  params
  ------
  off_sets : list of base shift for hamiltonians.
  boolean zw : 1 = zero worm.
  thres : value lower than thres reduce to 0. usually 1E-8;
  */
  void initial_setting(VD off_sets, double thres);

  size_t sps_sites(size_t i) { return _sps_sites[i]; }

  double shift() const { return origin_shift; }

  const std::vector<local_operator<MCT>> get_local_operators() const {
    return loperators;
  }

  const size_t num_types() { return num_type(site_type); }
};

extern template class model::base_model<bcl::heatbath>;
// extern template class model::base_model<bcl::st2010>;
extern template class model::base_model<bcl::st2013>;
