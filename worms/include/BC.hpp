#ifndef __BC__
#define __BC__
#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "model.hpp"

namespace BC{

  using TPROB = model::local_operator::TPROB;
  using VECD = std::vector<double>;

  std::vector<std::vector<double>> heatbath(std::vector<double> weights);

  bool check_probability_conservation(TPROB const&, double = 1.0e-10);
  bool check_detailed_balance(VECD const&, TPROB const&, double = 1.0e-10);
  bool check_balance_condition(VECD const&, TPROB const&, double = 1.0e-10);

  bool check(VECD const& weights, TPROB const& transition_matrix, double tolerance = 1.0e-10);

  //* create transition probability
  TPROB metropolis(VECD weights);
  TPROB st(VECD weights);

}

#endif