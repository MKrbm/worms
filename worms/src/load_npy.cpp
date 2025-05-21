// load_npy.cpp
#include <npy.hpp>
#include <iostream>
#include <complex>
#include <dirent.h>
#include <cstring>
#include <stdexcept>

// primary, real‐valued loader:
template<typename T>
std::pair<std::vector<unsigned long>, std::vector<T>>
load_npy(const std::string& path) {
    std::vector<unsigned long> shape;
    std::vector<T> data;
    bool fortran_order;
    try {
        npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    }
    catch(const std::exception& e) {
        std::cerr << "I/O error while reading npy file: "
                  << path << " : " << e.what() << "\n";
        std::exit(127);
    }
    return { shape, data };
}

// specialization for complex<double>:
template<>
std::pair<std::vector<unsigned long>, std::vector<std::complex<double>>>
load_npy<std::complex<double>>(const std::string& path) {
    std::vector<unsigned long> shape;
    bool fortran_order;

    // first try to read as complex<double> directly
    std::vector<std::complex<double>> cdata;
    try {
        npy::LoadArrayFromNumpy(path, shape, fortran_order, cdata);
        return { shape, cdata };
    }
    catch(...) {
      // if that fails, fall back to reading as real<double>
    }

    // fallback: load real data and zero‐pad imaginary parts
    std::vector<double> rdata;
    try {
        npy::LoadArrayFromNumpy(path, shape, fortran_order, rdata);
    }
    catch(const std::exception& e) {
        std::cerr << "I/O error while reading npy file (real fallback): "
                  << path << " : " << e.what() << "\n";
        std::exit(127);
    }

    cdata.resize(rdata.size());
    for (size_t i = 0; i < rdata.size(); ++i) {
        cdata[i] = std::complex<double>(rdata[i], 0.0);
    }
    return { shape, cdata };
}

// force instantiation of the primary template for double
template std::pair<std::vector<unsigned long>, std::vector<double>>
load_npy<double>(const std::string&);



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