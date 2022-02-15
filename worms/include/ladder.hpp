#pragma once
#include "model.hpp"
#include "load_npy.hpp"

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


  template <class MC>
  class ladder_v2 :public base_spin_model<1, 2, 4, MC>{
public:
  ladder_v2(std::vector<std::string> path_list, int L, double J1, double J2, double J3, double h, double shift=0, int pom=1);
  int L;
  const double J1,J2,J3,h;
  typedef base_spin_model<1, 2, 4, MC> MDT; 
  static lattice::graph return_lattice(int L){
    lattice::basis_t bs(1, 1);
    bs << 1;
    lattice::basis basis(bs);
    lattice::unitcell unitcell(1);
    unitcell.add_site(lattice::coordinate(0), 0);
    unitcell.add_bond(0, 0, lattice::offset(1), 0);
    lattice::span_t span(1,1); span << L;
    std::vector<lattice::boundary_t> boundary(1, lattice::boundary_t::periodic);
    lattice::graph lat(basis, unitcell, span, boundary);
    return lat;
  } 
};

}

model::ladder::ladder(int L, double J1, double J2, double J3, double h)
:L(L), J1(J1), J2(J2), J3(J3), 
h(h), base_spin_model(return_lattice(L))
{
  std::cout << "model output" << std::endl;
  std::cout << "  L : " << L<< std::endl;
  std::cout << "  [J1, J2, J3] : [" << J1 << 
       ", " << J2 << ", " << J3 << "]" << std::endl;
  
  std::cout << "  h : " << h << std::endl;
  std::cout << "  num local operators : " << Nop << std::endl;
  printf("  bond num : [type0, type1, typ3] = [%lu, %lu, %lu] \n", bond_t_size[0], bond_t_size[1], bond_t_size[2]);
  std::cout << "end \n" << std::endl;



  if (J1 < 0 || J2 < 0) std::cerr << "J1 and J2 must have non-negative value in this setting" << std::endl;

}

template <class MC>
model::ladder_v2<MC>::ladder_v2(std::vector<std::string> path_list, int L, double J1, double J2, double J3, double h, double shift, int pom)
:L(L), J1(J1), J2(J2), J3(J3), 
h(h), MDT(return_lattice(L))
{
  std::cout << "model output" << std::endl;
  std::cout << "  L : " << L<< std::endl;
  std::cout << "  [J1, J2, J3] : [" << J1 << 
       ", " << J2 << ", " << J3 << "]" << std::endl;
  
  std::cout << "  h : " << h << std::endl;
  std::cout << "  num local operators : " << MDT::Nop << std::endl;
  printf("  bond num : [type0] = [%lu] \n", MDT::bond_t_size[0]);
  std::cout << "end \n" << std::endl;



  if (J1 < 0 || J2 < 0) std::cerr << "J1 and J2 must have non-negative value in this setting" << std::endl;

  std::vector<double> off_sets(1,shift);
  auto& loperators = MDT::loperators;
  int l = 2;
  loperators[0] = local_operator<MC>(l, 2); 
  std::vector<double> J = {J1, (J2+J3)/2, (J2-J3)/2};
  MDT::leg_size[0] = l;

  for (int i=0; i<loperators[0].size; i++)
    for (int j=0; j<loperators[0].size; j++)  loperators[0].ham[j][i] = 0;
  

  int op_label = 0 ;
  for (auto path : path_list) {
    // std::cout << "ham read" << std::endl;
    auto pair = load_npy(path);
    auto shape = pair.first;
    auto data = pair.second;
    int l = 2;
    std::cout << "hamiltonian is read from " << path << std::endl;
    for (int i=0; i<shape[0]; i++){
      for (int j=0; j<shape[1]; j++)
      {
        auto x = J[op_label]*data[i * shape[1] + j];
        if (std::abs(x) > 1E-5) {
          loperators[0].ham[j][i] += x;
        }
      }
    }
    op_label++;
  }

  MDT::initial_setting(off_sets);  

  // std::cout << "hello" << std::endl;
  if (pom){
    for (int i=0; i<MDT::shifts.size(); i++){
      printf("shifts[%d] = %3.3f\n", i, MDT::shifts[i]);
    }
    for (int i=0; i<loperators[0].size; i++)
      for (int j=0; j<loperators[0].size; j++) if (std::abs(loperators[0].ham[j][i]) > 1E-5) {
          printf("[%2d, %2d] : %3.3f\n", j, i, loperators[0].ham[j][i]);
        }
  }
}