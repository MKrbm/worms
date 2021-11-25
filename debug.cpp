#include <uftree.hpp>
#include <model.hpp>
#include <worm.hpp>
#include <iostream>
#include <stdio.h>
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

    worm solver(0.6, h, 4);// If std < c++17, worm<heisenberg1D> instead.

    // solver.ops_sub = {
    //     {1,0},
    //     {2,2},
    //     {3,2},
    //     {5,2},
    //     {0,2},
    //     {4,2},
    //     {1,2},
    // };

    // for(int i = 0; i<solver.ops_sub.size(); i++){
    //     solver.ops_sub_tau.push_back((double)(i+1) / solver.ops_sub.size() * (solver.beta*0.9));
    // }

    solver.init_states();
    for (auto x : solver.state){
        printf("%d ", x);
    }
    solver.init_worm_rand();
    solver.init_front_group();
    solver.diagonal_update();

    for (auto x : solver.state){
        printf("%d ", x);
    }
    printf("\n");

    for (auto x : solver.ops_main){
        printf("bond : %d, optype : %d \n", x[0], x[1]);
    }

    printf("conn_op \n");
    int cnt = 0;
    for (auto x : solver.conn_op){
        printf("op: %d // ", cnt);
        for (int i=0; i<4; i++) printf("%d:%d ", i, x[i]);
        cnt++;
        printf("\n");
    }

    auto trans_prob = metropolis(h.weigths);


    return 0;
}