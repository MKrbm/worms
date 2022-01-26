#include <iostream>
#include <string>
#include <vector>
#include <ndimvec.hpp>
#include <load_npy.hpp>
// #include <Shastry.hpp>

using namespace std;


int main() {
  // auto ss = model::Shastry_2(2, 2, 1, 0);

  auto pair = load_npy("../python/array/H.npy");

  auto shape = pair.first;
  auto data = pair.second;

  printf("shape : [%lu, %lu]  \n", shape[0], shape[1]);

  printf("data : \n");

  for (int i=0; i<shape[0]; i++){
    std::cout << std::endl << "[ ";
    for (int j=0; j<shape[1]; j++)
    {
      // std::cout << data[i * shape[1] + j] << " ";
      printf("%2.1f ", data[i * shape[1] + j]);
    }
    std::cout << "]" << std::endl;
  }

  return 0;
}