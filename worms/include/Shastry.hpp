#pragma once
#include "model.hpp"
#include "load_npy.hpp"


namespace model{
  template <class MC>
  class Shastry :public base_spin_model<2, 4, MC>{
public:
  typedef base_spin_model<2,4,MC> MDT; 
  Shastry(int Lx, int Ly, double J1, double J2, double h); //(1) 
  Shastry(int L, double J1, double J2 = 1, double h = 0)
  :Shastry(L, L, J1, J2, h){}
  int Lx, Ly;
  int pom=0;
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

  template <class MC>
  class Shastry_2 :public base_spin_model<2, 4, MC>{
public:
    typedef base_spin_model<2, 4, MC> MDT; 
    Shastry_2(std::vector<std::string> path_list, int Lx, int Ly, double J1, double J2, double h, double s, int pom); //(1) 
    Shastry_2(std::vector<std::string> path_list, int L, double J1, double J2 = 1, double h = 0, double s = 0, int pom = 0)
    :Shastry_2(path_list, L, L, J1, J2, h, s, pom){}
    int Lx, Ly;
    int pom;
    const double J1,J2,h;
    static lattice::graph return_lattice(int Lx, int Ly){
      lattice::basis_t bs(2, 2); bs << 2, 0, 0, 2;
      lattice::basis basis(bs);
      lattice::unitcell unitcell(2);
      unitcell.add_site(lattice::coordinate(1/4.0, 1/4.0), 0);
      unitcell.add_site(lattice::coordinate(3/4.0, 3/4.0), 1);
      unitcell.add_bond(0, 1, lattice::offset(0, 0), 0);
      unitcell.add_bond(1, 0, lattice::offset(1, 0), 0);
      unitcell.add_bond(1, 0, lattice::offset(0, -1), 1);
      unitcell.add_bond(0, 1, lattice::offset(-1, -1), 1);

      lattice::span_t span(2, 2); span << Lx, 0, 0, Ly;
      std::vector<lattice::boundary_t> boundary(2, lattice::boundary_t::periodic);
      lattice::graph lat(basis, unitcell, span, boundary);
      // lat.num_bonds()
      return lat;
    } 
  };
}

template <class MC>
model::Shastry<MC>::Shastry(int Lx, int Ly, double J1, double J2, double h)
:Lx(Lx), Ly(Ly), J1(J1), J2(J2),
h(h), MDT(return_lattice(Lx, Ly))
{
std::cout << "model output" << std::endl;
std::cout << "[Lx, Ly] : [" << Lx << ", "<< Ly << "]" << std::endl;
std::cout << "[J1, J2] : [" << J1 << 
      ", " << J2 << "]" << std::endl;

std::cout << "h : " << h << std::endl;
std::cout << "num local operators : " << MDT::Nop << std::endl;
printf("bond num : [type0, type1] = [%lu, %lu] \n", MDT::bond_t_size[0], MDT::bond_t_size[1]);
std::cout << "end \n" << std::endl;

if (J1 < 0 || J2 < 0) std::cerr << "J1 and J2 must have non-negative value in this setting" << std::endl;



//* set bond operators. there are two types of such ops, depending on the coupling constant J1, J2.
int l = 2; //n*leg size (2 means operator is bond operator)
auto& loperators = MDT::loperators;
loperators[0] = local_operator<MC>(l); 
MDT::leg_size[0] = l;
loperators[1] = local_operator<MC>(l);
MDT::leg_size[1] = l;
//* bond b is assigined to i = bond_type(b) th local operator.

std::vector<double> off_sets(2,0);

//* setting for type 1 bond operator.
//* local unitary transformation is applied beforehand so that off-diagonal terms have non-negative value.
//* we ignore h for now.
// auto* loperator = &loperators[0];
loperators[0].ham[0][0] = -1/4.0;
loperators[0].ham[1][1] = 1/4.0;
loperators[0].ham[2][2] = 1/4.0;
loperators[0].ham[3][3] = -1/4.0;
loperators[0].ham[1][2] = 1/2.0;
loperators[0].ham[2][1] = 1/2.0;
for (auto& row:loperators[0].ham)
  for (auto& ele:row) ele *= J1;
off_sets[0] = 1/4.0;

//* setting for type 2
loperators[1].ham[0][0] = -1/4.0;
loperators[1].ham[1][1] = 1/4.0;
loperators[1].ham[2][2] = 1/4.0;
loperators[1].ham[3][3] = -1/4.0;
loperators[1].ham[1][2] = -1/2.0;
loperators[1].ham[2][1] = -1/2.0;
for (auto& row:loperators[1].ham)
  for (auto& ele:row) ele *= J2;
off_sets[1] = 1/4.0;


MDT::initial_setting(off_sets);
printf("local hamiltonian (type 1) / energy shift = %lf\n\n", MDT::shifts[0]);
for (int row=0; row<loperators[0].size; row++)
{
  for(int column=0; column<loperators[0].size; column++) 
    printf("%.2f   ", loperators[0].ham[row][column]);
  printf("\n");
}
printf("\n\n");

printf("local hamiltonian (type 2) / energy shift = %lf\n\n", MDT::shifts[1]);
for (int row=0; row<loperators[1].size; row++)
{
  for(int column=0; column<loperators[1].size; column++) 
    printf("%.2f   ", loperators[1].ham[row][column]);
  printf("\n");
}
printf("\n\n");

}

template <class MC>
model::Shastry_2<MC>::Shastry_2(std::vector<std::string> path_list, int Lx, int Ly, double J1, double J2, double h, double shift, int pom)
:Lx(Lx), Ly(Ly), J1(J1), J2(J2),pom(pom),
h(h), MDT(return_lattice(Lx, Ly))
{
std::cout << "\n\nmodel output" << std::endl;
std::cout << "[Lx, Ly] : [" << Lx << ", "<< Ly << "]" << std::endl;
std::cout << "[J1, J2] : [" << J1 << 
      ", " << J2 << "]" << std::endl;

std::cout << "h : " << h << std::endl;
std::cout << "num local operators : " << MDT::Nop << std::endl;
printf("bond num : [type0, type1] = [%lu, %lu] \n", MDT::bond_t_size[0], MDT::bond_t_size[1]);
std::cout << "end \n" << std::endl;

if (J1 < 0 || J2 < 0) std::cerr << "J1 and J2 must have non-negative value in this setting" << std::endl;


//* create sps vector.
std::vector<size_t> spsl(4, MDT::L);
MDT::set_sps(spsl);



//*set offset
std::vector<double> off_sets(2,shift);

//* read onsite hamiltonian
std::string os_path = path_list[2];
auto pair = load_npy(os_path);
auto shape_os = pair.first;
auto data_os = pair.second;

//* set loperators vector
auto& loperators = MDT::loperators;
int local = 0;
for (auto path : {path_list[0], path_list[1]}) {

  auto pair = load_npy(path);
  auto shape = pair.first;
  auto data = pair.second;
  int l = 2;
  loperators[local] = local_operator<MC>(l, 4); 
  MDT::leg_size[local] = l;
  std::cout << "hamiltonian is read from " << path << std::endl;



  for (int i=0; i<shape[0]; i++){
    for (int j=0; j<shape[1]; j++)
    {
      auto x = J1*data[i * shape[1] + j] + J2*data_os[i * shape[1] + j];
      if (std::abs(x) > 1E-4) {
        loperators[local].ham[j][i] = x;
        if (pom) printf("[%2d, %2d] : %3.3f\n", j, i, x);
        }
    }
  }
  if (pom)  std::cout << "\n\n" << std::endl;
  local ++;
}

MDT::initial_setting(off_sets);  

if (pom){
  for (int i=0; i<MDT::shifts.size(); i++){
    printf("shifts[%d] = %3.3f\n", i, MDT::shifts[i]);
  }
}

}

