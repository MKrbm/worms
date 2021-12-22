#pragma once
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <chrono>
#include <memory>
#include <iterator>
#include <cstdio>
#include <unistd.h>


template <unsigned int L = 2>
class binstate{
public:
  using base_type = long long unsigned int;
  static std::size_t const base_size;
  static std::size_t const size;
  // long long unsigned int x;
  base_type state[L/base_size + 1];
  binstate(){} 
  binstate(const base_type& x){
    state[0] = x;
    for(int i=1; i<size; i++) state[i] = 0;
  }

  unsigned int return_L(){
    return L;
  }

  binstate& operator=(base_type x){
    this->state[0] = x;
    return *this;
  }

  binstate& operator=(const binstate & bs){
    this->state = bs.state;
    return *this;
  }

  //* implement getbit
  unsigned int operator[](int p)
  {
    return (state[p/base_size] >> (p%base_size)) & 1;
  }

  //* implement AssiginBit
  void assign(int p, int b)
  {
    auto& s = state[p/base_size];
    p %= base_size;
    s = ((s & ~(1 << p)) | (b << p));
  }

  // //* implement flip
  void flip(int p)
  {
    auto& s = state[p/base_size];
    p %= base_size;
    s = s ^ (1<<p);
  }

  // binstate& operator=(int x){
  //   this->x = x;
  //   return *this;
  // }

  // binstate& operator=(int x){
  //   this->x = x;
  //   return *this;
  // }

  friend std::ostream& operator<<(std::ostream& os, const binstate& al)
  {
    // os << al.state;
    for (int i=al.size-1; i >= 0; i--)
    {
      os << al.state[i];
    }
    return os;
  }
  
};


template<typename binstate>
void flip(binstate& bs, int p, int base_size)
{
  auto& s = bs[p/base_size];
  p %= base_size;
  s = s ^ (1<<p);
}


//*getState
template<typename BINSTATE>
std::vector<int> getState(BINSTATE& bs)
{
  int L = bs.return_L();
  std::vector<int> state_(L);
  for (int i=0; i<L; i++) state_[i] = bs[i];
  return state_;
}

template<typename BINSTATE>
typename BINSTATE::base_type get_max_size(BINSTATE bs){
  return ((typename BINSTATE::base_type)1 << ((int)BINSTATE::base_size-1));
}

template <unsigned int L>
std::size_t const binstate<L>::base_size = 8*sizeof(long long unsigned int);

template <unsigned int L>
std::size_t const binstate<L>::size = L/base_size + 1;


namespace Binstate {
  template< int L = 2>
  using binarystate = long long unsigned int[L];
  

};