#include <chrono>
// #include <Shastry.hpp>
#include <ladder.hpp>
// #include "npy.hpp"

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

int main(){


  auto test = model::ladder(4,1,1,1,0);
  // test.lattice.print(std::cout);
  for (int i=0; i<test.bonds.size(); i++){
    printf("[%lu, %lu, %lu]\n", test.bonds[i][0], test.bonds[i][1], test.bond_type[i]);
  }







  
  return 0;
}