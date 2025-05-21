// load_npy.hpp
#pragma once
#include <utility>
#include <vector>
#include <string>
#include <complex>

// primary template: real‐valued only
template<typename T>
std::pair<std::vector<unsigned long>, std::vector<T>>
load_npy(const std::string& path);

// overload for convenience (defaults to double)
inline std::pair<std::vector<unsigned long>, std::vector<double>>
load_npy(const std::string& path) {
  return load_npy<double>(path);
}

// explicit specialization for std::complex<double>
template<>
std::pair<std::vector<unsigned long>, std::vector<std::complex<double>>>
load_npy<std::complex<double>>(const std::string& path);


//* get path to all .npy files
void get_npy_path(const std::string& dir_path, std::vector<std::string>& npy_path);
