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




  //latice
  lattice::graph lat = lattice::graph::simple(1,16);
  lat.print(std::cout);

  model::base_spin_model<> md(lat);


  spin_state::Operatorv2 op;
  typedef std::vector<int> veci;


  return 0;
}