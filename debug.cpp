#include <uftree.hpp>
#include <model.hpp>
#include <worm.hpp>
#include <iostream>
#include <stdio.h>
#include <state.hpp>
#include <chrono>
#include <memory>
#include <iterator>

using namespace std::chrono;

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
  worm solver(0.6, h, 1);

  solver.init_worms_rand();
  solver.init_states();
  int s = 1;
  for(auto&x : solver.state){
    x = s;
    s = 1^s;
  }


  solver.ops_sub.emplace_back(
  new model::OpState(
    {1,0,0,1},
    &solver.loperators[0],
    {2,3},
    0.01)
  );
  
  solver.ops_sub.emplace_back(
  new model::OpState(
    {1,0,0,1},
    &solver.loperators[0],
    {4,5},
    0.35)
  );

  solver.ops_sub.emplace_back(
  new model::OpState(
    {0,1,1,0},
    &solver.loperators[0],
    {4,5},
    0.4)
  );

  solver.ops_sub.emplace_back(
  new model::OpState(
    {0,1,1,0},
    &solver.loperators[0],
    {2,3},
    0.4)
  );

  // solver.set_dots(4, 0.33, 1, 0);
  // solver.set_dots(5, 0.33, 1, 1);

  solver.diagonal_update();
  // solver.ops_main.resize(0);
  // solver.state[2] = 0;
  solver.check_operators(solver.state, solver.ops_sub);
  solver.check_operators(solver.state, solver.ops_main);


  int ind = 0;
  for (auto&x : solver.spacetime_dots){
    cout << "index : " << ind;
    cout << "   site : " << x.site << endl; 
    cout << "type : " << model::return_name(x.dot_type, x.typeptr->is_off_diagonal()) << endl;
    
    cout << "tau : " << x.tau << endl;
    cout << "leg index : " <<x.typeptr->GetIndex(x.sptr, 0) << endl;


  


    printf("prev : %d, next : %d", x.prev, x.next);
    cout << "\n\n" ;
    ind++;
  }

  // solver.init_worms_rand();
  // solver.init_states();

  // solver.diagonal_update();
  // cout << "clear ops_main" << endl;
  // solver.ops_main.resize(0);

  // model::Worms& worms = *solver.pworms;
  // cout << "count of worms : " <<solver.pworms.use_count() << endl;


  // cout << "hi" << endl;
  // cout << "clear spacetime_dots" << endl;
  // solver.spacetime_dots.resize(4);
  // cout << "capacity : " << solver.spacetime_dots.capacity() << endl;
  // for (auto& x :  solver.spacetime_dots){
  //   cout << x.typeptr.use_count() << endl;
  // }







  // std::vector<model::OpStatePtr> ops_main;
  // std::vector<model::OpStatePtr> ops_sub;
  // model::WormsPtr pworms = model::WormsPtr(new model::Worms(3));

  // // auto& worms = pworms;

  // ops_main.push_back(model::OpStatePtr(new model::OpState(
  //   {-1,-1,-1,-1}, &h.loperators[0], {0,1}, {0,1}, 0
  // )));


  // cout << "integer representation of state : " << ops_main[0]->GetNum() << endl;


  // // ops_main.resize(0);


  // cout << "N_op for heisenberg1D is " << model::heisenberg1D::Nop << endl;

  // std::vector<model::Dot> dots;

  // dots.emplace_back(0, 0.1, 1, ops_main[0]->data(),
  //   ops_main[0], 1);

  // dots.emplace_back(0, 0.1, 1, pworms->data(),
  //   pworms, 1);

  // // printf("element : %d \n",*dot.sptr);
  // // printf("index : %ld \n",dot.typeptr->GetIndex(dot.sptr, 1));

  // // printf("num shared : %ld \n",dot.typeptr.use_count());


  // // // printf("num shared : %ld \n",dot.typeptr.use_count());



  // // ops_sub.push_back(ops_main[0]);
  // ops_main.clear();
  // dots.clear();
  // cout << "clear ops_main" << endl;
  // ops_sub.clear();
  // dot.~Dot();

  // printf("L : %d\n",ops_sub[0]->L);

  // printf("num shared : %ld \n",dot.typeptr.use_count());

  // printf("dot.site : %d\n", dot.site);



  // std::vector<std::shared_ptr<model::BaseState>> dots_main;
  // dots_main.push_back(std::shared_ptr<model::OpState>(new model::OpState(3, &h.loperators[0])));

  





  // model::local_operator ops(2);

  // ops.set_ham();

  // worm solver(0.6, h, 4);// If std < c++17, worm<heisenberg1D> instead.

  return 0;
}