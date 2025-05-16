// load_npy.hpp
#pragma once
#include <utility>
#include <vector>
#include <string>
#include <complex>

// -----------------------------------------------------------------
// Generic loader: T can be double or std::complex<double>
// -----------------------------------------------------------------
template<typename T>
std::pair<std::vector<unsigned long>, std::vector<T>>
load_npy(const std::string& path);

//* get path to all .npy files
void get_npy_path(const std::string& dir_path, std::vector<std::string>& npy_path);
