#pragma once
#include <utility>
#include <vector>
#include <string>
std::pair<std::vector<unsigned long>, std::vector<double>> load_npy(std::string path);

//* get path to all .npy files
void get_npy_path(std::string dir_path, std::vector<std::string>& npy_path);