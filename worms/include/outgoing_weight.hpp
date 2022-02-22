/*
   worms: a simple worm code

   Copyright (C) 2013-2021 by Synge Todo <wistaria@phys.s.u-tokyo.ac.jp>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#include <algorithm>
#include <vector>
// #include <state.hpp>

class outgoing_weight {
public:

  template<typename WEIGHT>
  outgoing_weight(WEIGHT const& w, int L, int sps)
    :L(L), sps(sps), pows(pows_array(sps)){ init_table(w); }
  outgoing_weight(int L, int sps)
    :L(L), sps(sps), pows(pows_array(sps)) {}
  template<typename WEIGHT>

  /*

  g : leg index where worm goes out
  l : l \in [1, .., 2**nls - 1] state of worm; local state change its state according to this. 

  e.g. in case of nls = 1.
  if l = 1 = [0,1] in bit representation and local state = 2 = [1,0]
  state -> [1,0] ^ [0,1] = [1,1] = 3.

  *params
  -------
  w : list of weights.
  boolean dw : 1 = have a chance to delete a worm while updating.
  */
  void init_table(WEIGHT const& w, bool dw = false){
    weights_.clear();
    weights_.resize(w.size());
    for (int s = 0; s < w.size(); ++s) {
      weights_[s].resize(2*L*(sps-1) + 1);
      for (int g = 0; g < 2*L; ++g){
        size_t t = pows[g+1];
        size_t a = pows[g];
        for (int l = 0; l < sps-1; l++) {
          size_t tmp = (s/t)*t + (s%t+(l+1)*a) % t;
          weights_[s][g*(sps-1) + l + 1] = w[tmp];
        }
        if (dw) weights_[s][0] = w[s];
        else weights_[s][0] = 0;
      }
    }
  }

  std::vector<size_t> pows_array(size_t sps = 2){
    std::vector<size_t> arr(2*L+1); size_t x = 1;
    for (int i=0; i<2*L+1; i++) {
      arr[i]=x; x*=sps;
    };
    return arr;
  }
  std::vector<double> const& operator[](int s) { return weights_[s]; }
  int size(){return weights_.size();}
private:
  std::vector<std::vector<double> > weights_;
  int L;
  std::size_t sps; //onsite Hilbert space dimension.
  std::vector<size_t> pows;
};
