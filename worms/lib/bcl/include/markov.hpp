/*
   Copyright (C) 2009-2021 by Synge Todo <wistaria@phys.s.u-tokyo.ac.jp>,
                              Hidemaro Suwa <suwamaro@phys.s.u-tokyo.ac.jp>

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

#include <random>
#include "random_choice.hpp"

namespace bcl {

template<typename CutoffType>
class markov_impl {
private:
  typedef random_choice<CutoffType> rc_type;
public:
  typedef std::size_t state_type;
  markov_impl() {}
  markov_impl(std::vector<std::vector<double> > const& tm) {
    init(tm);
  }
  template<typename MC, typename WVEC>
  markov_impl(MC const&, WVEC const& weights) {
    std::vector<std::vector<double> > tm;
    MC::generate_transition_matrix_resize(weights, tm);
    init(tm);
  }
  void init(std::vector<std::vector<double> > const& tm) {
    rc_.clear();
    for(std::size_t i = 0; i < tm.size(); ++i) rc_.push_back(rc_type(tm[i]));
  }
  template<typename ENGINE>
  state_type operator()(state_type prev, ENGINE& eng) {
    return rc_[prev](eng);
  }
private:
  std::size_t dim;
  std::vector<rc_type> rc_;
};

template<typename ENGINE>
class markov : public markov_impl<typename ENGINE::result_type> {
private:
  typedef markov_impl<typename ENGINE::result_type> base_type;
public:
  markov() : base_type() {}
  markov(std::vector<std::vector<double> > const& tm) : base_type(tm) {}
  template<typename MC, typename WVEC>
  markov(MC const& mc, WVEC const& weights) : base_type(mc, weights) {}
};


template<typename CutoffType>
class sparse_markov_impl {
private:
  typedef random_choice<CutoffType> rc_type;
  std::vector<long long> sparse_idx;
  std::vector<size_t> original_idx;
public:
  typedef std::size_t state_type;
  sparse_markov_impl() {}
  template<typename MC, typename WVEC>
  sparse_markov_impl(MC const&, WVEC const& weights, std::vector<long long> sparse_idx, std::vector<size_t> original_idx)
  : sparse_idx(sparse_idx), original_idx(original_idx)
  {
    std::vector<std::vector<double> > tm;
    MC::generate_transition_matrix_resize(weights, tm);
    init(tm);
  }
  void init(std::vector<std::vector<double> > const& tm) {
    rc_.clear();
    for(std::size_t i = 0; i < tm.size(); ++i) rc_.push_back(rc_type(tm[i]));
  }
  template<typename ENGINE>
  state_type operator()(state_type prev, ENGINE& eng) {
    return original_idx[rc_[sparse_idx[prev]](eng)];
  }
private:
  std::size_t dim;
  std::vector<rc_type> rc_;
};


template<typename ENGINE>
class sparse_markov : public sparse_markov_impl<typename ENGINE::result_type> {
private:
  typedef sparse_markov_impl<typename ENGINE::result_type> base_type;
public:
  sparse_markov() : base_type() {}

  /*
  params
  ------
  sparse_idx : original_idx to sparse_idx, if given original_idx is pointing at zero elemtn, then it returns -1.
  original_idx : sparse_idx to original_idx.
  */
  template<typename MC, typename WVEC>
  sparse_markov(MC const& mc, WVEC const& weights, std::vector<long long> sparse_idx, std::vector<size_t> original_idx) : base_type(mc, weights, sparse_idx, original_idx){}
};

} // end namespace bcl
