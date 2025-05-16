#include <npy.hpp>
#include <iostream>
#include <complex>
#include <dirent.h>
#include <cstring>
#include <stdexcept>

// -----------------------------------------------------------------
// Template definition (you can also inline this in the header)
// -----------------------------------------------------------------
template<typename T>
std::pair<std::vector<unsigned long>, std::vector<T>>
load_npy(const std::string& path) {
    std::vector<unsigned long> shape;
    std::vector<T> data;
    bool fortran_order;
    try {
        // this will automatically pick the right overload
        npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    }
    catch(const std::exception& e) {
        std::cerr << "I/O error while reading npy file : "
                  << path << " : " << e.what() << "\n";
        std::exit(127);
    }
    return { shape, data };
}

// Explicit instantiations so linker can pick them up:
template std::pair<std::vector<unsigned long>, std::vector<double>>
    load_npy<double>(const std::string& path);

template std::pair<std::vector<unsigned long>, std::vector<std::complex<double>>>
    load_npy<std::complex<double>>(const std::string& path);


// -----------------------------------------------------------------
// Directory traversal (unchanged)
// -----------------------------------------------------------------
void get_npy_path(const std::string& dir_path, std::vector<std::string>& npy_path) {
    DIR *di = opendir(dir_path.c_str());
    if (!di) {
        throw std::runtime_error("cannot open folder : " + dir_path);
    }
    while (auto *dir = readdir(di)) {
        // split on last dot instead of strtok (safer)
        std::string name(dir->d_name);
        auto pos = name.rfind('.');
        if (pos != std::string::npos && name.substr(pos) == ".npy") {
            npy_path.push_back(dir_path + "/" + name);
        }
    }
    closedir(di);
}