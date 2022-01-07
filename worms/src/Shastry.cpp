#include "../include/Shastry.hpp"

model::Shastry::Shastry(int Lx, int Ly, double J1, double J2, double h)
:Lx(Lx), Ly(Ly), J1(J1), J2(J2),
h(h), base_spin_model(return_lattice(Lx, Ly))
{
  std::cout << "model output" << std::endl;
  std::cout << "[Lx, Ly] : [" << Lx << ", "<< Ly << "]" << std::endl;
  std::cout << "[J1, J2] : [" << J1 << 
       ", " << J2 << "]" << std::endl;
  
  std::cout << "h : " << h << std::endl;
  std::cout << "num local operators : " << Nop << std::endl;
  printf("bond num : [type0, type1] = [%lu, %lu] \n", bond_t_size[0], bond_t_size[1]);
  std::cout << "end \n" << std::endl;

  if (J1 < 0 || J2 < 0) std::cerr << "J1 and J2 must have non-negative value in this setting" << std::endl;



  //* set bond operators. there are two types of such ops, depending on the coupling constant J1, J2.
  int l = 2; //n*leg size (2 means operator is bond operator)
  loperators[0] = local_operator(l); 
  leg_size[0] = l;
  loperators[1] = local_operator(l);
  leg_size[1] = l;
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

  
  initial_setting(off_sets);
  printf("local hamiltonian (type 1) / energy shift = %lf\n\n", shifts[0]);
  for (int row=0; row<loperators[0].size; row++)
  {
    for(int column=0; column<loperators[0].size; column++) 
      printf("%.2f   ", loperators[0].ham[row][column]);
    printf("\n");
  }
  printf("\n\n");

  printf("local hamiltonian (type 2) / energy shift = %lf\n\n", shifts[1]);
  for (int row=0; row<loperators[1].size; row++)
  {
    for(int column=0; column<loperators[1].size; column++) 
      printf("%.2f   ", loperators[1].ham[row][column]);
    printf("\n");
  }
  printf("\n\n");

}


model::Shastry_2::Shastry_2(int Lx, int Ly, double J1, double J2, double h)
:Lx(Lx), Ly(Ly), J1(J1), J2(J2),
h(h), base_spin_model(return_lattice(Lx, Ly))
{
  std::cout << "model output" << std::endl;
  std::cout << "[Lx, Ly] : [" << Lx << ", "<< Ly << "]" << std::endl;
  std::cout << "[J1, J2] : [" << J1 << 
       ", " << J2 << "]" << std::endl;
  
  std::cout << "h : " << h << std::endl;
  std::cout << "num local operators : " << Nop << std::endl;
  printf("bond num : [type0, type1] = [%lu, %lu] \n", bond_t_size[0], bond_t_size[1]);
  std::cout << "end \n" << std::endl;

  if (J1 < 0 || J2 < 0) std::cerr << "J1 and J2 must have non-negative value in this setting" << std::endl;



  //* set bond operators. there are two types of such ops, depending on the coupling constant J1, J2.
  int l = 2; //n*leg size (2 means operator is bond operator)
  loperators[0] = local_operator(l); 
  leg_size[0] = l;
  //* bond b is assigined to i = bond_type(b) th local operator.

  std::vector<double> off_sets(1,0);

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


  
  initial_setting(off_sets);
  printf("local hamiltonian (type 1) / energy shift = %lf\n\n", shifts[0]);
  for (int row=0; row<loperators[0].size; row++)
  {
    for(int column=0; column<loperators[0].size; column++) 
      printf("%.2f   ", loperators[0].ham[row][column]);
    printf("\n");
  }
  printf("\n\n");

}
