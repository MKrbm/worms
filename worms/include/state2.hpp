#pragma once

#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
// #include "model.hpp"
#include "automodel.hpp"

using std::cout;
using std::endl;

using namespace model;

namespace spin_state{
  class Dotv2;
  class OpStatev2;
  class Wormsv2;
  
  using US = unsigned short;
  using VUS = vector<US>;
  using WORM = std::tuple<int, int, double>;
  using WORM_ARR = std::vector<WORM>; //  site, spin, dot_label, tau (dot label is needed for reverse lookup)
  using DOT_ARR = std::vector<std::tuple<int,int,int,int>>;   //prev, next, dot_type, index, (index refers to the legs of the dot with respect to the class of dots)

  struct StateFunc{
  public:
    const VS pows;
    const size_t sps;
    const size_t leg_size;
    StateFunc(size_t, size_t);
    static VS pows_array(size_t, size_t);
    size_t state2num(VUS const&, int = -1);
    size_t state2num(VUS const&, VS const&);
    VUS num2state(int, int);
  };

  // STATE num2state(int num, int L);
  // std::string return_name(int dot_type, int op_type);


  static std::string op_type_name[2] = {
      "diagonal",
      "off-diagonal"
    };


  static std::string dot_type_name[3] = {
    "state",
    "operator",
    "worm"
  };


}


/*
params
------
int prev : previous dot label
int% sptr : ptr to state
int dot_type : state type where -1 : state, -2 : worms, non-negative integer : operator label. 
int index : index which will be used to indexing the corresponding type list. e.g. if dot_type = -1. state[index] is the spin on the dot, if -2, worm[index] is the worm corresponds to the dot. However, if dot_type=0, wich means the dot is operator, ops[dot_type] is the operator of the dot and the index refers to the position of dot with respect to the operator.
*/
class spin_state::Dotv2
{
  size_t prev_;
  size_t next_;
  int dot_type_;
  size_t index_; 
  size_t site_;
public:
  Dotv2(){}
  Dotv2(size_t p, size_t n, int o, size_t i, size_t s)
  :prev_(p), next_(n), dot_type_(o), index_(i), site_(s)
  {}

  static Dotv2 state(size_t s) { return Dotv2(s, s, -1, s, s); }
  static Dotv2 worm(size_t p, size_t n, size_t wl, size_t s) { return Dotv2(p, n, -2, wl, s); }
  size_t prev() const { return prev_; }
  size_t next() const { return next_; }
  size_t site() const {return site_;}
  size_t leg(size_t dir, size_t L) const {
    if (at_operator()) return dir*L + index_;
    else return 0;
  }
  size_t label() const {
    if (at_operator()) return dot_type_;
    else return index_;
  }

  bool at_operator() const { return dot_type_ >= 0; }
  bool at_origin() const { return dot_type_ == -1; }
  bool at_worm() const { return dot_type_ == -2; }
  void set_prev(size_t p) { prev_ = p; }
  void set_next(size_t n) { next_ = n; }
  size_t move_next(size_t dir) const {
    return (dir == 0) ? prev_ : next_;
    ASSERT(false, "dir can be 1 or 0");
  }
};


class spin_state::Wormsv2{
  size_t site_;
  size_t spin_;
  size_t dot_label_;
  double tau_;
public:
  Wormsv2(){}
  Wormsv2(size_t si, size_t sp, size_t dl, double t):site_(si), spin_(sp),dot_label_(dl),tau_(t)
  {}
  
  void set_spin(int s) { spin_ = s; }
  size_t site() const {return site_;}
  size_t spin()const {return spin_;}
  size_t dot_label()const {return dot_label_;}
  double tau()const {return tau_;}
};


// template <size_t sps_, size_t max_L>
// const std::array<size_t, max_L+1> spin_state::Operatorv2<sps_, max_L>::pows = pows_array<max_L>(sps_);