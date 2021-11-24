#ifndef __model__
#define __model__

#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <array>

//$\hat{H} = \sum_{<i,j>} [J \vec{S}_i \dot \vec{S}_j - h/Nb (S_i^z + S_j^z)]$ 
// map spin to binary number e.g. -1 \rightarrow 0, 1 \rightarrow 1
template <int N_op, int N_Dop>
// N_op : number of operators
// N_Dop : number of diagonal operators
class base_model_spin_1D{
  using CONN = std::array<int,4>; // holds connection of an operator. 0 = [0, 0] / 1 = [0, 1] / 2 = [1, 0] / 3 = [1, 1]
public:
  int L;
  int Nb;
  int NDop = N_Dop;
  int Nop = N_op;
  double rho = 0;
  base_model_spin_1D(int L_, int Nb_, int N_op_)
  :L(L_), Nb(Nb_){}

  std::array<CONN, N_op> operator_list;
  std::array<double, N_op> weigths; // weight of each operator.
  std::array<double, N_Dop> prob; 
  std::array<std::array<int, N_op>, N_op> worm_dir; // 0 : go reverse, 1 : reverse direction & change site, 2 : go straight, 3 : go straight & change site.

};


class heisenberg1D :public base_model_spin_1D<3,2>{
public:
  heisenberg1D(int L, double Jz, double Jxy, double h, bool PBC = true); //(1) 
  heisenberg1D(int L, double h, double Jz=1, bool PBC = true) : heisenberg1D(L, Jz, -Jz, h, PBC) {} //(2) : pass arguments to (1) constructor. This is for AFH.
  const static int N_op;
  static std::vector<std::vector<int>> return_bonds(int L, bool PBC);
  const std::vector<std::vector<int>> bonds;
  double Jz, Jxy;
  const int L; //systemsize of hesenbergmodel
  const bool PBC;
  const int Nb;
  const double h;


};

#endif 