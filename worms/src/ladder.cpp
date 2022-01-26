#include "../include/ladder.hpp"
#include "../include/testmodel.hpp"
#include "../include/load_npy.hpp"

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