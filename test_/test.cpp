// #include <chrono>
// #include <Shastry.hpp>
#include <ladder.hpp>
// #include "npy.hpp"
#include <iostream>
#include <chrono>
#include <array>

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

std::array<size_t, 5> pows = {1, 4, 16, 4*16, 16*16};

size_t update_state_1(size_t s, size_t leg, size_t fl=1){
  size_t t = pows[leg+1];
  size_t a = pows[leg];
  return (s/t)*t + (s%t+fl*a) % t;
}

size_t update_state_2(size_t s, size_t leg, size_t fl=1){
  return s ^ (fl << (2*leg)); 
}

int main(){


  auto test = model::ladder(4,1,1,1,0);
  // test.lattice.print(std::cout);
  for (int i=0; i<test.bonds.size(); i++){
    printf("[%lu, %lu, %lu]\n", test.bonds[i][0], test.bonds[i][1], test.bond_type[i]);
  }


  // size_t sum_=0;
  // size_t s = 10;
  // auto t1 = high_resolution_clock::now();
  // cout << update_state_1(s, 0, 0) << endl;
  // cout << update_state_1(s, 0, 1) << endl;
  // cout << update_state_1(s, 0, 2) << endl;
  // cout << update_state_1(s, 0, 3) << endl;
  // cout << update_state_1(s, 1, 1) << endl;
  // cout << update_state_1(s, 1, 2) << endl;
  // cout << update_state_1(s, 1, 3) << endl;
  // cout << update_state_1(s, 2, 1) << endl;
  // cout << update_state_1(s, 2, 2) << endl;
  // cout << update_state_1(s, 2, 3) << endl;
  // cout << update_state_1(s, 3, 1) << endl;



  // auto t2 = high_resolution_clock::now();
  // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  // double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / (double)1E3;
  // std::cout << sum_ << std::endl;
  // cout << "Elapsed : " << elapsed << endl;










  
  return 0;
}