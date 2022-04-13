#pragma once
#include "model.hpp"

using std::cout;
using std::endl;

namespace model{
  template <class MC>
  class MG :public base_spin_model<2, 2, 4, MC>{
public:
    typedef base_spin_model<2,2,4,MC> MDT; 
    MG(std::vector<std::string> path_list, int L, double s = 0, int pom = 0);
    int L;
    int pom=0;
    double sft = 0;
    static lattice::graph return_lattice(int L){
      lattice::basis_t bs(1, 1);
      bs << 1;
      lattice::basis basis(bs);
      lattice::unitcell unitcell(1);
      unitcell.add_site(lattice::coordinate(0), 0);
      unitcell.add_bond(0, 0, lattice::offset(1), 0);
      unitcell.add_bond(0, 0, lattice::offset(2), 1);
      lattice::span_t span(1,1); span << L;
      std::vector<lattice::boundary_t> boundary(1, lattice::boundary_t::periodic);
      lattice::graph lat(basis, unitcell, span, boundary);
      return lat;
    } 
  };

  template <class MC>
  class MG_2 :public base_spin_model<1, 8, 4, MC>{
public:
    typedef base_spin_model<1,8,4,MC> MDT; 
    MG_2(std::vector<std::string> path_list, int L, double s = 0, int pom = 0);
    int L;
    int pom=0;
    double sft = 0;
    static lattice::graph return_lattice(int L){
      ASSERT(L%3 == 0, "L must be multiple of 3");
      L = L/3;
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


template <class MC>
model::MG<MC>::MG(
  std::vector<std::string> path_list, int L, 
  double s, int pom )
:L(L), sft(s), pom(pom), MDT(return_lattice(L))
{
  cout << "model output" << endl; 
  cout << "L : " << L << endl;

  cout << "num local operators : " << MDT::Nop << endl;
  printf("bond num : [type0, type1] = [%lu, %lu] \n", MDT::bond_t_size[0], MDT::bond_t_size[1]);
  cout << "end \n" << endl;

  ASSERT(path_list.size() == 1, "size of pathlist is 1 for MG");


  auto& loperators = MDT::loperators;
  auto& leg_size = MDT::leg_size;
  leg_size[0] = 2;
  leg_size[1] = 2;
  std::vector<double> J = {1, 1/2.0};
  std::vector<std::string> path_list_ = { path_list[0], path_list[0]};
  std::vector<size_t> type_list = {0, 1};
  double thres = 1E-8;
  set_hamiltonian<MDT::Nop, MDT::max_sps, MDT::max_L, typename MDT::MCT>(
    loperators,
    leg_size,
    path_list_,
    type_list,
    J);
  std::vector<double> off_sets(MDT::Nop,sft);
  MDT::initial_setting(off_sets, thres, true);  
  if (pom){
    for (int i=0; i<MDT::shifts.size(); i++){
      printf("shifts[%d] = %3.3f\n", i, MDT::shifts[i]);
    }
  }
}


template <class MC>
model::MG_2<MC>::MG_2(
  std::vector<std::string> path_list, int L, 
  double s, int pom )
:L(L), sft(s), pom(pom), MDT(return_lattice(L))
{

  ASSERT(L%3 == 0, "L must be multiple of 3 in order to use MG_2 model");
  cout << "model output" << endl; 
  cout << "L : " << L << endl;

  cout << "num local operators : " << MDT::Nop << endl;
  cout << "end \n" << endl;

  ASSERT(path_list.size() == 1, "size of pathlist is 1 for MG");


  auto& loperators = MDT::loperators;
  auto& leg_size = MDT::leg_size;
  leg_size[0] = 2;
  std::vector<double> J = {1};
  std::vector<std::string> path_list_ = { path_list[0]};
  std::vector<size_t> type_list = {0};
  double thres = 1E-8;
  set_hamiltonian<MDT::Nop, MDT::max_sps, MDT::max_L, typename MDT::MCT>(
    loperators,
    leg_size,
    path_list_,
    type_list,
    J);
  std::vector<double> off_sets(MDT::Nop,sft);
  MDT::initial_setting(off_sets, thres, true);  
  if (pom){
    for (int i=0; i<MDT::shifts.size(); i++){
      printf("shifts[%d] = %3.3f\n", i, MDT::shifts[i]);
    }
  }
}
