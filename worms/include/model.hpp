#ifndef __model__
#define __model__

#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <array>
#include <math.h>


#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif


namespace model {
    class local_operator;

    
    typedef std::vector<int> STATE;

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
        std::array<std::array<int, N_op>, N_op> worm_dir; // 0 : go back to the same root, 1 : reverse direction & change site, 2 : go straight, 3 : go straight & change site.

        std::array<std::array<double, N_op>, N_op> trans_prob;

    };

    class heisenberg1D;


    int state2num(STATE state, int L);
    STATE num2state(int num, int L);



}

class model::local_operator{

public:
    int L; // number of site operator acts 
    int size; // size of operator (2**L)
    std::vector<std::vector<double>> ham;
    std::vector<double> ham_vector;
    std::vector<std::tuple<int, double>> trans_prob;

    local_operator(int L)
    :L(L), size(pow(2, L))
    {
        ham = std::vector<std::vector<double>>(size, std::vector<double>(size, 0));
        ham_vector = std::vector<double>(size*size, 0);
        trans_prob = std::vector<std::tuple<int, double>>(size*size);
    }

    void set_ham(){
      int N = ham_vector.size();

      for (int i=0; i<N; i++){
        auto index = num2index(i);
        ham_vector[i] = ham[index[0]][index[1]];
      }
    }

    std::array<int, 2> num2index(int num){
      ASSERT(num < size*size, "num is invalid");
      std::array<int, 2> index;
      index[0] = num%size;
      index[1] = num/size;
      return index;
    }

    int index2num(std::array<int, 2> index){
      ASSERT(index[0] < size && index[1] < size, "index is invalid");
      int num = 0;
      num += index[0];
      num += index[1] * size;
      return num;
    }
};



class model::heisenberg1D :public model::base_model_spin_1D<3,2>{
public:
    heisenberg1D(int L, double Jz, double Jxy, double h, bool PBC = true); //(1) 
    heisenberg1D(int L, double h, double Jz=1, bool PBC = true) : heisenberg1D(L, Jz, -Jz, h, PBC) {} //(2) : pass arguments to (1) constructor. This is for AFH.
    const static int N_op;
    local_operator loperator;
    static std::vector<std::vector<int>> return_bonds(int L, bool PBC);
    const std::vector<std::vector<int>> bonds;
    double Jz, Jxy;
    const int L; //systemsize of hesenbergmodel
    const bool PBC;
    const int Nb;
    const double h;

    int DopAtRand(double);

};

#endif 