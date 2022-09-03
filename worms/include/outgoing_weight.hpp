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
  template<typename WEIGHT>
  std::vector<double> init_table(WEIGHT const& w, size_t s, bool dw = false){
    // weights_.resize(w.size());
    weights_.clear();
    weights_.resize(2*L*(sps-1) + 1);
    for (int g = 0; g < 2*L; ++g){
      size_t t = pows[g+1];
      size_t a = pows[g];
      for (int l = 1; l < sps; l++) {
        size_t tmp = (s/t)*t + (s%t+l*a) % t;
        weights_[g*(sps-1) + l] = w[tmp];
      }
      if (dw) weights_[0] = w[s];
      else weights_[0] = 0;
    }
    return weights_;
    // }
  }
  template<typename WEIGHT>
  auto init_table_sparse(WEIGHT const& w, size_t s, bool zw = false){
    // weights_.resize(w.size());
    sparse_idx.clear();
    sparse_idx.resize(w.size(), -1);
    original_idx.resize(0);
    weights_.resize(0);
    size_t s_idx=0;

    double zero_weight;
    zero_weight = zw ? w[s] : 0;

    if (zero_weight != 0){
      weights_.push_back(zero_weight);
      sparse_idx[0] = s_idx++;
      original_idx.push_back(0);
    }


    for (int g = 0; g < 2*L; ++g){
      size_t t = pows[g+1];
      size_t a = pows[g];
      for (int l = 1; l < sps; l++) {
        size_t tmp = (s/t)*t + (s%t+l*a) % t;
        if (w[tmp] != 0){
          weights_.push_back(w[tmp]);
          sparse_idx[g*(sps-1) + l] = s_idx++;
          original_idx.push_back(g*(sps-1) + l);
        }
      }
    }
    return make_tuple(weights_, sparse_idx, original_idx);
    // }
  }

  std::vector<size_t> pows_array(size_t sps = 2){
    std::vector<size_t> arr(2*L+1); size_t x = 1;
    for (int i=0; i<2*L+1; i++) {
      arr[i]=x; x*=sps;
    };
    return arr;
  }

  // std::vector<double> const& operator[](int s) { return weights_[s]; }
  // int size(){return weights_.size();}
  std::vector<double> weights_;
  std::vector<long long> sparse_idx;
  std::vector<size_t> original_idx;
  size_t L;
  std::size_t sps; //onsite Hilbert space dimension.
  std::vector<size_t> pows;
};
