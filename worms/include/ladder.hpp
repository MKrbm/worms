#pragma once
#include "model.hpp"

namespace model{
  class ladder :public base_spin_model<3>{
public:
  ladder(int L, double J1, double J2, double J3, double h); //(1) 
  int L;
  const double J1,J2,J3,h;
  static lattice::graph return_lattice(int L){
    lattice::basis_t bs(2, 2);
    bs << 1,0,0,2;
    lattice::basis basis(bs);
    lattice::unitcell unitcell(2);
    unitcell.add_site(lattice::coordinate(0, 0), 0);
    unitcell.add_site(lattice::coordinate(0, 1/2.0), 0);
    unitcell.add_bond(0, 0, lattice::offset(1, 0), 0);
    unitcell.add_bond(1, 1, lattice::offset(1, 0), 0);
    unitcell.add_bond(0, 1, lattice::offset(0, 0), 1);
    unitcell.add_bond(0, 1, lattice::offset(1, 0), 2);
    unitcell.add_bond(1, 0, lattice::offset(1, 0), 2);
    lattice::span_t span(2, 2); span << L, 0, 0, 1;
    std::vector<lattice::boundary_t> boundary = {
      lattice::boundary_t::periodic, lattice::boundary_t::open
    };
    lattice::graph lat(basis, unitcell, span, boundary);
    return lat;
  } 
};

}

