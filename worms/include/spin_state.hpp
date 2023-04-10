#pragma once

#include <iostream>
#include <vector>


namespace spin_state{
  using VS = std::vector<size_t>;
  using VVS = std::vector<VS>;
  using VI = std::vector<int>;
  using VVI = std::vector<VI>;
  using VD = std::vector<double>;
  using US = unsigned short;
  using VUS = std::vector<US>;
  using UC = unsigned char;
  using VUC = std::vector<UC>;
  using spin_t = US;
  using state_t = std::vector<spin_t>;
}