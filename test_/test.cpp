// #include <chrono>
// #include <Shastry.hpp>
#include <ladder.hpp>
// #include "npy.hpp"
#include <iostream>
#include <chrono>
#include <array>
#include <type_traits>
#include <string>
#include <vector>

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

template <class, template <class> class>
struct is_instance : public std::false_type {};

template <class T, template <class> class U>
struct is_instance<U<T>, U> : public std::true_type {};

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

  // using A = model::Shastry_2<bcl::st2010>;
  // using B = model::Shastry_2<void>;

  // std::cout << is_instance<A, model::Shastry_2>::value << std::endl;
  std::vector<std::string> path_list = std::vector<std::string>({
    "../python/array/lad_bond_ori0.npy",
    "../python/array/lad_bond_ori1.npy",
    "../python/array/lad_bond_ori2.npy",
  });
  auto test = model::ladder_v2<bcl::st2010>(path_list, 4,1,1,1,0);
  // test.lattice.print(std::cout);
  // for (int i=0; i<test.bonds.size(); i++){
  //   printf("[%lu, %lu, %lu]\n", test.bonds[i][0], test.bonds[i][1], test.bond_type[i]);
  // }
  size_t sum_=0;
  size_t s = 10;
  auto t1 = high_resolution_clock::now();
  // for (int i=0; i<(int)1E7; i++){
  //   for (int j=0; j<4; j++){
  //     s = update_state_2(s, j, ((3+i)%4));
  //     sum_ += s;
  //     // cout << sum_ << endl;
  //     // if ((update_state_1(s, j, (i%4))) != (update_state_2(s, j, (i%4)))) printf("%lu, %lu, %d\n", s, j, (i%4));
  //   }
  //   sum_ %= 999;
  // // }
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