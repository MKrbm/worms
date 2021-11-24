#include <uftree.hpp>
#include <model.hpp>
#include <worm.hpp>
#include <iostream>
#include <BC.hpp>

#define DEBUG 0

using std::cout;
using std::endl;

int add(int a, int b){
    return a+b;
}

int main(){
    std::cout << "debug start" << std::endl;
    std::mt19937 rand_src(12345);
    heisenberg1D h(6,1,1);

    worm<heisenberg1D> solver(1.0, h, 10);// If std < c++17, worm<heisenberg1D> instead.
    solver.init_worm_rand();
    solver.init_front_group();
    solver.diagonal_update();

    auto trans_prob = metropolis(h.weigths);


    return 0;
}