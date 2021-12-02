#ifndef __BC__
#define __BC__
#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>

std::vector<std::vector<double>> heatbath(std::vector<double> weights);

template<typename Arr, int n = Arr{}.size()>
std::array<std::array<double, n>, n> metropolis(Arr weights){
  std::array<std::array<double, n>, n> trans_prob;
  int S = weights.size();
  std::array<double, n> tmp;
  double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  double add;
  for(int i=0; i<S; i++){
    double d = 1/(weights[i] * (S-1));
    tmp[i] = 1;
    for(int j=1; j<S; j++){
      add = std::min(weights[i], weights[(i+j)%S]) * d;
      tmp[(i+j)%S] = add;
      tmp[i] -= add;
    }
    trans_prob[i] = tmp;
  }

  return trans_prob;
}

#endif