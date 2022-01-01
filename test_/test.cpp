#include <uftree.hpp>
#include <model.hpp>
#include <worm.hpp>
#include <observable.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <state.hpp>
#include <chrono>
#include <memory>
#include <iterator>
#include <cstdio>
#include <tuple>
#include <binstate.hpp>
#include <lattice/graph.hpp>
#include <lattice/coloring.hpp>
#include <Shastry.hpp>

using namespace std::chrono;

#define DEBUG 0

using std::cout;
using std::endl;
using std::ofstream;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;
int add(int a, int b){
  return a+b;
}

inline int modifyBit(int n, int p, int b)
{
    return ((n & ~(1 << p)) | (b << p));
}

inline int getbit(int n, int p)
{
    return (n >> p) & 1;
}

int main(){


  auto ss = model::Shastry(2, 2, 1);

  // //latice
  // // lattice::graph lat = lattice::graph::simple(1,16);
  // lattice::basis_t bs(2, 2); bs << 2, 0, 0, 2;
  // lattice::basis basis(bs);
  // lattice::unitcell unitcell(2);
  // unitcell.add_site(lattice::coordinate(0, 0), 1);
  // unitcell.add_site(lattice::coordinate(0, 1.0/2), 1);
  // unitcell.add_site(lattice::coordinate(1.0/2, 0), 0);
  // unitcell.add_site(lattice::coordinate(1.0/2, 1.0/2), 0);
  // unitcell.add_bond(0, 1, lattice::offset(0, 0), 0);
  // unitcell.add_bond(0, 2, lattice::offset(0, 0), 0);
  // unitcell.add_bond(1, 0, lattice::offset(0, 1), 0);
  // unitcell.add_bond(2, 0, lattice::offset(1, 0), 0);
  // unitcell.add_bond(2, 3, lattice::offset(0, 0), 0);
  // unitcell.add_bond(1, 3, lattice::offset(0, 0), 0);
  // unitcell.add_bond(3, 2, lattice::offset(0, 1), 0);
  // unitcell.add_bond(3, 1, lattice::offset(1, 0), 0);
  // unitcell.add_bond(0, 3, lattice::offset(0, 0), 1);
  // unitcell.add_bond(1, 2, lattice::offset(-1, -1), 1);





  // lattice::span_t span(2, 2); span << 2, 0, 0, 2;
  // std::vector<lattice::boundary_t> boundary(2, lattice::boundary_t::periodic);
  // lattice::graph lat(basis, unitcell, span, boundary);
  // lat.print(std::cout);
  // auto color = lattice::coloring(lat);

  // lattice::basis_t bs(1, 1); bs << 2; // 1x1 matrix
  // lattice::basis basis(bs);
  // lattice::unitcell unitcell(1);
  // unitcell.add_site(lattice::coordinate(0), 0);
  // unitcell.add_site(lattice::coordinate(0.5), 0);
  // unitcell.add_bond(0, 1, lattice::offset(0), 0);
  // unitcell.add_bond(1, 0, lattice::offset(1), 0);
  // lattice::span_t span(1, 1); span << 8; // 1x1 matrix
  // std::vector<lattice::boundary_t> boundary(1, lattice::boundary_t::periodic);
  // lattice::graph lat(basis, unitcell, span, boundary);
  // lat.print(std::cout);
  // std::cout << "bond_type of 0 is : " << lat.bond_type(0) << std::endl;;

  
  return 0;
}