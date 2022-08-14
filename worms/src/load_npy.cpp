#include "../include/load_npy.hpp"
#include <npy.hpp>


std::pair<std::vector<unsigned long>, std::vector<double>> load_npy(std::string path) {
  using namespace std;
  vector<unsigned long> shape;
  bool fortran_order;
  vector<double> data;

  shape.clear();
  data.clear();
  npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
  return std::make_pair(shape, data);
}