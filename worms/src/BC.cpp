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

BC::TPROB BC::st(VECD weights){
  int n = weights.size();
  TPROB tm(n, VECD(n, 0.0));

  using std::abs;
  std::vector<double> accum(n+1, 0);
  double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (sum==0){
    for (int i=0; i<n; i++) tm[i][i] = 1;
    return tm;
  }
  double shift = *(std::max_element(weights.begin(), weights.end())) / sum;
  accum[0] = 0;
  for (std::size_t i = 0; i < n; ++i) accum[i+1] = accum[i] + weights[i] / sum;

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      tm[i][j] = (std::max(std::min(accum[i+1] + shift, accum[j+1]) -
                            std::max(accum[i] + shift, accum[j]), 0.0) +
                  std::max(std::min(accum[i+1] + shift, accum[j+1] + 1) -
                            std::max(accum[i] + shift, accum[j] + 1), 0.0));
    }
  }
  // for (std::size_t i = 0; i < n; ++i) {
  //   for (std::size_t j = i; j < n; ++j) {
  //     double t = (tm[i][j] + tm[j][i]) / 2;
  //     tm[i][j] = t / (weights[i] / sum);
  //     tm[j][i] = t / (weights[j] / sum);

  //   }
  // }
  for (std::size_t i = 0; i < n; ++i) {
    if (weights[i] > 0){
      for (std::size_t j = 0; j < n; ++j) {
        tm[i][j] /= (weights[i] / sum);
        // tm[j][i] = t / (weights[j] / sum);
      }
    }else{
      for (std::size_t j = 0; j < n; ++j) {
        tm[i][j] = (weights[j] / sum);
        // tm[j][i] = t / (weights[j] / sum);
      }
    }
  }
  return tm;
}



bool BC::check_probability_conservation(TPROB const& transition_matrix, double tolerance) {
  std::size_t n = transition_matrix.size();
  for (std::size_t i = 0; i < n; ++i) {
    double p = 0;
    for (std::size_t j = 0; j < n; ++j) p += transition_matrix[i][j];
    if (std::abs(p - 1) > tolerance) return false;
  }

  return true;
}


bool BC::check_balance_condition(VECD const& weights, TPROB const& transition_matrix, double tolerance) {
  typedef typename VECD::value_type value_type;
  std::size_t n = weights.size();
  for (std::size_t j = 0; j < n; ++j) {
    value_type w = 0;
    for (std::size_t i = 0; i < n; ++i) w += weights[i] * transition_matrix[i][j];
    if (std::abs(w - weights[j]) > tolerance * weights[j]) return false;
  }
  return true;
}


bool BC::check_detailed_balance(VECD const& weights, TPROB const& transition_matrix,double tolerance) {
  typedef typename VECD::value_type value_type;
  value_type sum = std::accumulate(weights.begin(), weights.end(), value_type(0));
  std::size_t n = weights.size();
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      if (std::abs(weights[i] * transition_matrix[i][j] - weights[j] * transition_matrix[j][i]) >
          tolerance * sum) return false;
  return true;
}

bool BC::check(VECD const& weights, TPROB const& transition_matrix, double tolerance) {
  return check_probability_conservation(transition_matrix, tolerance) &&
    check_balance_condition(weights, transition_matrix, tolerance);
  }