#include <iostream>
#include <vector>
using namespace std;

#pragma once


template <class VEC>
void print_vec(VEC vec){
  cout << vec.size() << endl;
  for (auto x : vec){
    cout << x << endl;
  }
}


extern template void print_vec<vector<int>>(vector<int> vec);
extern template void print_vec<vector<float>>(vector<float> vec);
extern template void print_vec<vector<double>>(vector<double> vec);
