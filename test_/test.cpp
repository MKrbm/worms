#include <chrono>
// #include <Shastry.hpp>
#include <testmodel.hpp>
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


  // auto ss = model::Shastry_2(2, 2, 1, 0);
  auto test = model::test(3);

  test.lattice.print(std::cout);

  // std::cout << "bonds : \n" ;
  // size_t ind;
  // for (auto x : ss.bonds){
  //   printf("[ %d, %d ] / type : %d\n", x[0], x[1], ss.bond_type[ind]);
  //   ind++;

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