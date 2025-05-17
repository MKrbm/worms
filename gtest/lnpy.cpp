#include "gtest/gtest.h"
#include "load_npy.hpp"
#include <npy.hpp>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <complex>

namespace fs = std::filesystem;

// To print stdout with ctest, you need to:
// 1. Run ctest with the -V (verbose) flag: ctest -V
// 2. Or use GTEST_COUT instead of std::cout as shown below

// Define a macro that redirects to std::cerr which is not buffered
#define GTEST_COUT std::cerr << "[          ] "

TEST(LoadNpyTest, RealArrayLoad) {
    // Write a small real-valued array to a temporary .npy file
    fs::path tmp = fs::temp_directory_path() / "test_real.npy";
    std::vector<unsigned long> shape = {2, 3};
    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    GTEST_COUT << "Real array tmp path: " << tmp.string() << std::endl;
    npy::SaveArrayAsNumpy(tmp.string(), false, shape.size(), shape.data(), data);

    // Load it via the wrapper overload (defaults to double)
    auto [loaded_shape, loaded_data] = load_npy<double>(tmp.string());
    EXPECT_EQ(loaded_shape, shape);
    EXPECT_EQ(loaded_data, data);

    fs::remove(tmp);
}

TEST(LoadNpyTest, ComplexArrayLoad) {
    // Write a small complex-valued array to a temporary .npy file
    fs::path tmp = fs::temp_directory_path() / "test_cplx.npy";
    std::vector<unsigned long> shape = {2, 2};
    std::vector<std::complex<double>> data = {{1, -1}, {2, -2}, {3, -3}, {4, -4}};
    GTEST_COUT << "Complex array tmp path: " << tmp.string() << std::endl;
    npy::SaveArrayAsNumpy(tmp.string(), false, shape.size(), shape.data(), data);

    // Load it via explicit template parameter
    auto [loaded_shape, loaded_data] = load_npy<std::complex<double>>(tmp.string());
    EXPECT_EQ(loaded_shape, shape);
    EXPECT_EQ(loaded_data, data);

    fs::remove(tmp);
}
