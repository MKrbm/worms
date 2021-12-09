#include "../include/BC.hpp"

std::vector<std::vector<double>> heatbath(std::vector<double> weights){
  std::vector<std::vector<double>> trans_prob;
  int S = weights.size();
  std::vector<double> tmp(S,0);
  double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  for(int i=0; i<S; i++)
  
    for(int j=0; j<S; j++){
      tmp[j] = weights[j] / sum;
    }
    trans_prob.push_back(tmp);

  return trans_prob;
}


BC::TPROB BC::metropolis(VECD weights){
  int n = weights.size();
  TPROB tm(n, VECD(n));

  for (std::size_t i = 0; i < n; ++i) {
    tm[i][i] = 1;
    if (weights[i] > 0) {
      for (std::size_t j = 0; j < n; ++j) {
        if (i != j) {
          tm[i][j] = std::min(weights[i], weights[j]) / weights[i] / (n-1);
          tm[i][i] -= tm[i][j];
        }
      }
      tm[i][i] = abs(tm[i][i]);
    } else {
      for (std::size_t j = 0; j < n; ++j)
        if (i != j) tm[i][j] = 0;
    }
  }
  return tm;
}