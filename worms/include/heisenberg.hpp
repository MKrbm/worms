#pragma once
#include "model.hpp"


namespace model{
  template <class MC>
  class heisenberg :public base_spin_model<1, 1, 4, MC>{
  public:
    typedef base_spin_model<1, 1, 4, MC> MDT; 
    heisenberg(int L, double Jz, double Jxy, double h, int dim); //(1) 
    heisenberg(int L, double h, int dim = 1, double Jz=1) : heisenberg(L, Jz, -Jz, h, dim) {} //(2) : pass arguments to (1) constructor. This is for AFH.

    double Jz, Jxy;
    const double h;
    int dim;
  };
}



template <class MC>
model::heisenberg<MC>::heisenberg(int L, double Jz, double Jxy, double h, int dim)
  :Jz(Jz), Jxy(Jxy), dim(dim),
  h(h), MDT(lattice::graph::simple(dim, L))
{
  std::cout << "model output" << std::endl;
  std::cout << "L : " << L << std::endl;
  std::cout << "Nb : " << MDT::Nb << std::endl;
  std::cout << "Jz : " << Jz << std::endl;
  std::cout << "h : " << h << std::endl;
  std::cout << "end \n" << std::endl;

  int l = 2;
  auto& loperators = MDT::loperators;
  loperators[0] = local_operator<MC>(l);
  MDT::leg_size[0] = l;
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
      printf("%.2f   ", MDT::loperators[0].ham[row][column]);}
   
    printf("\n");
  }

  std::vector<double> off_sets(MDT::Nop, off_set);
  MDT::initial_setting(off_sets); //set_hum will be called inside the function


  // printf("\n\nprint trans weights : \n\n");
  // loperator.print_trans_weights();
  printf("end setting\n\n\n");

  MDT::rho = loperators[0].max_diagonal_weight_ * MDT::Nb;
}
