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

