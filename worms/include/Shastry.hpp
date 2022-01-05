#pragma once
#include "model.hpp"

namespace model{
  class Shastry :public base_spin_model<2>{
public:
  Shastry(int Lx, int Ly, double J1, double J2 = 1, double h = 0); //(1) 
  Shastry(int L, double J1, double J2 = 1, double h = 0)
  :Shastry(L, L, J1, J2, h){}
  int Lx, Ly;
  const double J1,J2,h;
  static lattice::graph return_lattice(int Lx, int Ly){
    lattice::basis_t bs(2, 2); bs << 2, 0, 0, 2;
    lattice::basis basis(bs);
    lattice::unitcell unitcell(2);
    unitcell.add_site(lattice::coordinate(0, 0), 0);
    unitcell.add_site(lattice::coordinate(0, 1.0/2), 0);
    unitcell.add_site(lattice::coordinate(1.0/2, 0), 0);
    unitcell.add_site(lattice::coordinate(1.0/2, 1.0/2), 0);
    unitcell.add_bond(0, 1, lattice::offset(0, 0), 0);
    unitcell.add_bond(0, 2, lattice::offset(0, 0), 0);
    unitcell.add_bond(1, 0, lattice::offset(0, 1), 0);
    unitcell.add_bond(2, 0, lattice::offset(1, 0), 0);
    unitcell.add_bond(2, 3, lattice::offset(0, 0), 0);
    unitcell.add_bond(1, 3, lattice::offset(0, 0), 0);
    unitcell.add_bond(3, 2, lattice::offset(0, 1), 0);
    unitcell.add_bond(3, 1, lattice::offset(1, 0), 0);
    unitcell.add_bond(0, 3, lattice::offset(0, 0), 1);
    unitcell.add_bond(1, 2, lattice::offset(-1, -1), 1);
    lattice::span_t span(2, 2); span << Lx, 0, 0, Ly;
    std::vector<lattice::boundary_t> boundary(2, lattice::boundary_t::periodic);
    lattice::graph lat(basis, unitcell, span, boundary);
    // lat.num_bonds()
    return lat;
  } 
};

}

