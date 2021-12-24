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

#define DEBUG 0

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
  double beta = 1;
  double h = 1;
  BC::observable ene; // energy 
  BC::observable umag; // uniform magnetization 


  std::mt19937 rand_src(12345);
  model::heisenberg1D h1(L,h,J);
  worm solver(beta, h1);

  /*
  * test for swap functions
  std::vector<spin_state::OpStatePtr> ops1(1E3, 
    spin_state::OpStatePtr(new spin_state::OpState(
    {1,0,0,1},
    &solver.loperators[0],
    {2,3},
    0.01)));

  std::vector<spin_state::OpStatePtr> ops2(1E3, 
    spin_state::OpStatePtr(new spin_state::OpState(
    {1,0,0,1},
    &solver.loperators[0],
    {4,5},
    0.01)));


  auto t1 = high_resolution_clock::now();
  for (int i=0; i<1E4; i++){
    // auto tmp = ops1;
    // ops1 = ops2;
    // ops1 = tmp;
    ops1.swap(ops2);
  }
  auto t2 = high_resolution_clock::now();
  double elapsed = duration_cast<milliseconds>(t2 - t1).count() / (double)1E3;

  cout << "elapsed time : " << elapsed << endl;
  */
  

  //* state (using vector)
  std::vector<int> state(6,1); 
  std::vector<int> lstate(2);

  auto t1 = high_resolution_clock::now();
  for(std::size_t i=0; i<1E6; i++){
    int s1 = (i*100)%6;
    int s2 = (i*99)%6;
    int s3 = (i*98)%6;
    int s4 = (i*97)%6;
    int s5 = (i*96)%6;

    state[s1] = !state[s1];
    state[s2] = 1;
    state[s3] = 0;
    lstate[0] = state[s4];
    lstate[1] = state[s5];
  }
  auto t2 = high_resolution_clock::now();
  double elapsed = duration_cast<milliseconds>(t2 - t1).count() / (double)1E3;

  cout << "elapsed time : " << elapsed << endl;

  cout << "state[0] = " << state[0] << endl;

  //* state (using integer)
  unsigned long state_ = ~0;
  unsigned int lstate_ = ~0;
  binstate<6> X; 
  binstate<2> LX; 
  t1 = high_resolution_clock::now();
  std::array<long long unsigned int, 1> XX = {0};
  int size = X.base_size;

  int c = 0;
  for(std::size_t i=0; i<1E6; i++){
    int s1 = (i*100)%6;
    int s2 = (i*99)%6;
    int s3 = (i*98)%6;
    int s4 = (i*97)%6;
    int s5 = (i*96)%6;

    state_ = state_ ^ (1<<s1);
    state_ = modifyBit(s2, state_, 1);
    state_ = modifyBit(s3, state_, 0);
    lstate_ = modifyBit(0, lstate_, getbit(state_, s4)); 
    lstate_ = modifyBit(1, lstate_, getbit(state_, s5)); 
    // c += state_;
    // X.flip(s1);
    // flip(XX, s1, size);
    // X.assign(s2, 1);
    // X.assign(s3, 0);
    // LX.assign(0, X[s4]);
    // LX.assign(1, X[s5]);

  }


  t2 = high_resolution_clock::now();
  elapsed = duration_cast<milliseconds>(t2 - t1).count() / (double)1E3;
  cout << "elapsed time : " << elapsed << endl;


  // auto state1 = getState(X);
  // for (auto s : XX) cout << s << " ";
  // cout << endl;
  cout << c << endl;

  return 0;
}