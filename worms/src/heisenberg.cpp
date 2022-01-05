#include "../include/BC.hpp"
#include "../include/heisenberg.hpp"




model::heisenberg::heisenberg(int L, double Jz, double Jxy, double h, int dim)
  :Jz(Jz), Jxy(Jxy), dim(dim),
  h(h), base_spin_model(lattice::graph::simple(dim, L))
{
  std::cout << "model output" << std::endl;
  std::cout << "L : " << L << std::endl;
  std::cout << "Nb : " << Nb << std::endl;
  std::cout << "Jz : " << Jz << std::endl;
  std::cout << "h : " << h << std::endl;
  std::cout << "end \n" << std::endl;

  int l = 2;
  loperators[0] = local_operator(l);
  leg_size[0] = l;
  auto& loperator = loperators[0];
  // set hamiltonian
  // loperator.ham[0][0] = h;
  // auto color = lattice::coloring(lat);
  int is_bip = 1;
  double off_set = 0;
  if (L%2!=0) {
    is_bip = -1;
    off_set = 1.0/4;
  }
  loperator.ham[0][0] = -1/4.0 + h/2.0;
  loperator.ham[1][1] = 1/4.0;
  loperator.ham[2][2] = 1/4.0;
  loperator.ham[3][3] = -1/4.0 - h/2.0;
  loperator.ham[1][2] = is_bip * 1/2.0;
  loperator.ham[2][1] = is_bip * 1/2.0;
  //end


  
  printf("set local hamiltonian : \n\n");
  for (int row=0; row<loperator.size; row++)
  {
    for(int column=0; column<loperator.size; column++)
    {
      printf("%.2f   ", loperator.ham[row][column]);}
   
    printf("\n");
  }

  std::vector<double> off_sets(Nop, off_set);
  initial_setting(off_sets); //set_hum will be called inside the function


  // printf("\n\nprint trans weights : \n\n");
  // loperator.print_trans_weights();
  printf("end setting\n\n\n");

  rho = loperators[0].max_diagonal_weight_ * Nb;
}



