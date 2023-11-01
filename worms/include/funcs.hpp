#pragma once
#ifndef NDEBUG
#define printd(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define dout cout
#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__         \
                << " line " << __LINE__ << ": " << message << std::endl;       \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)
#else
#define ASSERT(condition, message)                                             \
  do {                                                                         \
  } while (false)
#define printd(fmt, ...) 0
#define dout 0 && cout
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <execinfo.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[ ";
  for (const T &item : vec)
    os << item << ", ";
  os << "]";
  return os;
}

void fillStringWithSpaces(std::string &str, int targetLength);

std::string getCurrentDateTime();

class Logger {
public:
  Logger(const std::string &filename) : outfile(filename) {
    // check if file was opened successfully
    if (!outfile.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      exit(1);
    }
  }

  template <typename T> Logger &operator<<(const T &value) {
    std::cout << value;
    outfile << value;
    return *this;
  }

  // Overload for std::endl
  Logger &operator<<(std::ostream &(*pf)(std::ostream &)) {
    std::cout << pf;
    outfile << pf;
    return *this;
  }

private:
  std::ofstream outfile;
};
