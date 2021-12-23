#include <uftree.hpp>
#include <model.hpp>
#include <worm.hpp>
#include <observable.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <state.hpp>
#include <chrono>
#include <memory>
#include <iterator>
#include <cstdio>
#include <binstate.hpp>

using namespace std::chrono;


using std::cout;
using std::endl;
using std::ofstream;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;
int add(int a, int b){
  return a+b;
}

inline int modifyBit(int n, int p, int b)
{
    return ((n & ~(1 << p)) | (b << p));
}

inline int getbit(int n, int p)
{
    return (n >> p) & 1;
}

int main(){



  std::cout << "debug start" << std::endl;
  int L = 6;
  double J = 1;
  double beta = 5;
  double h = 0;
  BC::observable ene; // energy 
  BC::observable umag; // uniform magnetization 


  std::mt19937 rand_src(12345);
  model::heisenberg1D h1(L,h,J);
  worm solver(beta, h1);
  cout << "initialized " << endl;

  solver.init_states();
  solver.ops_sub.resize(0);
  solver.diagonal_update(3);
  // solver.check_operators(solver.state, solver.ops_sub);
  // solver.check_operators(solver.state, solver.ops_main);
  // solver.worm_update();
  // solver.swap_oplist();
  // solver.diagonal_update();

  // int n_kink=0;
  // for (int i=0; i < 1E3; i++){
  //   solver.init_states();
  //   solver.ops_sub.resize(0);
  //   for (int i=0; i< 5*1E2; i++){
  //     solver.diagonal_update();
  //     solver.check_operators(solver.state, solver.ops_sub);
  //     solver.check_operators(solver.state, solver.ops_main);
  //     solver.worm_update();
  //     solver.swap_oplist();
  //   }
  //   n_kink += solver.ops_sub.size();
  // }

  // double energy = -1.5 * ((double)n_kink/MCSTEP) / beta;




  // int ind = 0;
  // ofstream outputfile("logs.txt");
  // outputfile  << "stdout is redirected to a file\n\n\n"; // this is written to redir.txt
  // for (auto&x : solver.spacetime_dots){
  //   // cout << spin_state::return_name(x.dot_type, x.typeptr->is_off_diagonal()) <<endl;
  //   outputfile << "index : " << ind;
  //   outputfile << "   site : " << x.site << endl; 
  //   outputfile << "type : " << spin_state::return_name(x.dot_type, x.typeptr->is_off_diagonal()) << endl;
    
  //   outputfile << "tau : " << x.tau << endl;
  //   outputfile << "leg index : " <<x.typeptr->GetIndex(x.sptr, 0) << endl;
  //   // printf("prev : %d, next : %d", x.prev, x.next);
  //   outputfile << "prev : " << x.prev << ", next : " << x.next << endl;
  //   outputfile << "\n\n" ;
  //   ind++;
  // }

  // outputfile << "\n\noutput state : " << endl;
  // for (auto&x : solver.state){
  //   outputfile << x << " ";
  // }
  // outputfile.close();


  // cout << "number of operators : " << solver.ops_main.size() << endl;

  return 0;
}