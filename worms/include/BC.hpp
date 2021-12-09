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

  TPROB metropolis(VECD weights);
}

#endif