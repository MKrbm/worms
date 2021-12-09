#include <uftree.hpp>
#include <model.hpp>
#include <worm.hpp>
#include <iostream>
#include <stdio.h>

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





    // model::local_operator ops(2);

    // ops.set_ham();

    // worm solver(0.6, h, 4);// If std < c++17, worm<heisenberg1D> instead.

    return 0;
}