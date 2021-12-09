#pragma once

#include "model.hpp"

namespace model{
  class LState;
}

class model::LState : public std::vector<int>
{
  typedef std::vector<int> vec;
  public :
  const local_operator* plop;
  LState():plop(nullptr){
    std::cout << "what happend" << std::endl;
  }
  ~LState(){
      // std::cout << "Call Destructor (Radius : " << size() << ")" << std::endl;

  }
  LState(int L, local_operator* plop):vec(L, -1), plop(plop){}
};