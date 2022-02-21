#pragma once
#include "model.hpp"
#include "load_npy.hpp"


namespace model{
  template <class MC>
  class heisenberg :public base_spin_model<1, 2, 4, MC>{
  public:
    typedef base_spin_model<1, 2, 4, MC> MDT; 
    heisenberg(int L, double Jz, double Jxy, double h, int dim); //(1) 
    heisenberg(int L, double h, int dim = 1, double Jz=1) : heisenberg(L, Jz, -Jz, h, dim) {} //(2) : pass arguments to (1) constructor. This is for AFH.

    double Jz, Jxy;
    const double h;
    int dim;
  };


  template <class MC>
  class heisenberg_v2 :public base_spin_model<1, 2, 4, MC>{
  public:
    typedef base_spin_model<1, 2, 4, MC> MDT; 
    heisenberg_v2(std::vector<std::string> path_list, int L, double Jz, double Jxy, double h, int dim, double shift=0, int pom = 1); //(1) 
    heisenberg_v2(std::vector<std::string> path_list, int L, double h, int dim = 1, double Jz=1, double shift=0, int pom = 1) : heisenberg_v2(path_list, L, Jz, -Jz, h, dim, shift, pom) {} //(2) : pass arguments to (1) constructor. This is for AFH.

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




template <class MC>
model::heisenberg_v2<MC>::heisenberg_v2(std::vector<std::string> path_list, int L, double Jz, double Jxy, double h, int dim, double shift, int pom)
  :Jz(Jz), Jxy(Jxy), dim(dim),
  h(h), MDT(lattice::graph::simple(dim, L))
{
  std::cout << "model output" << std::endl;
  std::cout << "L : " << L << std::endl;
  std::cout << "Nb : " << MDT::Nb << std::endl;
  std::cout << "  [Jz, Jx, Jy] : [" << Jz << 
       ", " << Jxy << ", " << Jxy << "]" << std::endl;
  std::cout << "h : " << h << std::endl;
  std::cout << "end \n" << std::endl;
  std::vector<double> J = {Jz, Jxy, Jxy, h/(2*dim)};

  auto& loperators = MDT::loperators;
  auto& leg_size = MDT::leg_size;
  leg_size[0] = 2; //there is only one type of operator.
  auto type_list = std::vector<size_t>(path_list.size(), 0); // size is equal to N_op

  set_hamiltonian<MDT::Nop, MDT::max_sps, MDT::max_L, typename MDT::MCT>(
    loperators,
    leg_size,
    path_list,
    type_list,
    J
  );
  // int l = 2;
  // loperators[0] = local_operator<MC>(l);
  // leg_size[0] = l;
  // auto& loperator = loperators[0];
  // int is_bip = 1;
  // for (int i=0; i<loperators[0].size; i++)
  //   for (int j=0; j<loperators[0].size; j++)  loperators[0].ham[j][i] = 0;
  // //end

  // int op_label = 0 ;
  // for (auto path : path_list) {
  //   auto pair = load_npy(path);
  //   auto shape = pair.first;
  //   auto data = pair.second;
  //   int l = 2;
  //   std::cout << "hamiltonian is read from " << path << std::endl;
  //   for (int i=0; i<shape[0]; i++){
  //     for (int j=0; j<shape[1]; j++)
  //     {
  //       auto x = J[op_label]*data[i * shape[1] + j];
  //       if (std::abs(x) > 1E-5) {
  //         loperators[0].ham[j][i] += x;
  //       }
  //     }
  //   }
  //   op_label++;
  // }
  std::vector<double> off_sets(MDT::Nop, shift);
  MDT::initial_setting(off_sets);
  if (pom){
    for (int i=0; i<MDT::shifts.size(); i++){
      printf("shifts[%d] = %3.3f\n", i, MDT::shifts[i]);
    }
    for (int i=0; i<loperators[0].size; i++)
      for (int j=0; j<loperators[0].size; j++) if (std::abs(loperators[0].ham[j][i]) > 1E-5) {
          printf("[%2d, %2d] : %3.5f\n", j, i, loperators[0].ham[j][i]);
        }
  }

}
