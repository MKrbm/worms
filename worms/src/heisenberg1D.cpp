#include "../include/BC.hpp"
#include "../include/model.hpp"

std::vector<std::vector<int>> model::heisenberg1D::return_bonds(int L, bool PBC){
  std::vector<std::vector<int>> bonds(L, std::vector<int>(2));

  for(int i=0; i<L; i++){
    bonds[i][0] = i;
    bonds[i][1] = (i+1)%L;
  }
  // std::vector<std::vector<int>> vtr {{34,55},{45},{53,62}};
  return bonds;
}


model::heisenberg1D::heisenberg1D(int L, double Jz, double Jxy, double h, bool PBC)
  :Jz(Jz), Jxy(Jxy),
  h(h), base_model_spin_1D(L, PBC ? L : L-1, PBC, return_bonds(L,PBC))
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
  loperator.ham[0][0] = -1/4.0 + h/2.0;
  loperator.ham[1][1] = 1/4.0;
  loperator.ham[2][2] = 1/4.0;
  loperator.ham[3][3] = -1/4.0 - h/2.0;
  loperator.ham[1][2] = 1/2.0;
  loperator.ham[2][1] = 1/2.0;
  //end


  
  printf("set local hamiltonian : \n\n");
  for (int row=0; row<loperator.size; row++)
  {
    for(int column=0; column<loperator.size; column++)
    {
      printf("%.2f   ", loperator.ham[row][column]);}
   
    printf("\n");
  }
  initial_setting(); //set_hum will be called inside the function


  // printf("\n\nprint trans weights : \n\n");
  // loperator.print_trans_weights();
  printf("end setting\n\n\n");

  rho = loperators[0].total_weights * Nb;
}

void model::heisenberg1D::initial_setting(){

  int i = 0;
  double tmp=0;
  for (auto& x : loperators){
    x.set_ham();
    tmp += x.total_weights;
    operator_cum_weights[i] = tmp;
    shifts.push_back(x.ene_shift);
  }
}

