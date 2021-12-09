#include <uftree.hpp>
#include <model.hpp>
#include <worm.hpp>
#include <iostream>
#include <stdio.h>
#include <state.hpp>

#define DEBUG 0

using std::cout;
using std::endl;

int add(int a, int b){
  return a+b;
}

int main(){
  std::cout << "debug start" << std::endl;
  std::mt19937 rand_src(12345);
  model::heisenberg1D h(6,1,1);


  std::vector<model::LState> lstates(1E4, model::LState(1E5, &h.loperators[0]));

  std::cout << "memory is allocating" << endl;

  std::cout << "deleting memory" << endl;
  lstates.resize(0);
  std::cout << "finish deleting" << endl;



  std::vector<std::vector<int>> lstates1(1E4, std::vector<int>(1E5, -1));

  std::cout << "memory is allocating" << endl;

  std::cout << "deleting memory" << endl;
  lstates1.resize(0);
  std::cout << "finish deleting" << endl;



  // model::local_operator ops(2);

  // ops.set_ham();

  // worm solver(0.6, h, 4);// If std < c++17, worm<heisenberg1D> instead.

  return 0;
}