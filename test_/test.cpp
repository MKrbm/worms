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
#include <tuple>
#include <binstate.hpp>
#include <lattice/graph.hpp>
#include <lattice/coloring.hpp>
#include <Shastry.hpp>
#include <npy.hpp>

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


  auto ss = model::Shastry_2(2, 2, 1, 0);
  ss.lattice.print(std::cout);

  // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


  // int c = 0;
  // for (int i = 0; i < (int)1E6; i++){
  //   int tmp = i % 100;
  //   // c+= (tmp << (i%10));
  //   c += tmp * std::pow((int)2, (int)(i%10));
  //   // c %= (int)(1E6);
  // }


  // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
  // std::cout << " c = " << c << std::endl;
  // std::cout << "elapsed : " << elapsed << std::endl;





  
  return 0;
}