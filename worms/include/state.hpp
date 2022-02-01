#pragma once

#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
#include "model.hpp"

using std::cout;
using std::endl;


namespace spin_state{
  class Dotv2;
  class OpStatev2;
  class Wormsv2;
  template <size_t nls_=1, size_t max_L = 4>
  class Operatorv2;
  
  using SPIN = model::SPIN;
  using size_t = std::size_t; 
  using STATE = model::STATE;
  using BOND = model::BOND;
  using local_operator = model::local_operator;
  using WORM = std::tuple<int, int, double>;
  using WORM_ARR = std::vector<WORM>; //  site, spin, dot_label, tau (dot label is needed for reverse lookup)
  using DOT_ARR = std::vector<std::tuple<int,int,int,int>>;   //prev, next, dot_type, index, (index refers to the legs of the dot with respect to the class of dots)

  template<size_t nls> 
  struct state_func{
  public:
    size_t nls_ = nls;
    static const size_t sps = (1<<nls);
    static size_t state2num(STATE const& state, int L = -1){
      // std::cout << "nls = " << nls << std::endl;
      size_t num = 0;
      if (L < 0) L = state.size();
      if (L == 0) return 0;
      for (int i = L-1; i >= 0; i--) {
        num <<= (nls);
        num += state[i];
      }
      return num;
    }

    static size_t state2num(STATE const& state, BOND const& bond){
      size_t u = 0;
      for (int i=0; i<bond.size(); i++){
        // int tmp = cstate[bond[i]];
        u += (state[bond[i]] << (nls*i));
      }
      return u;
    }


    static STATE num2state(int num, int L){
      int coef = 1;
      model::STATE state(L, 0); // all spin up
      for (int i=0; i<L; i++){
        state[i] = num&(sps-1);
        num >>= (nls);
      }
      return state;
    }
  };

  inline size_t state2num(STATE const& state, int L = -1){
    size_t num = 0;
    if (L < 0) L = state.size();
    if (L == 0) return 0;
    for (int i = L-1; i >= 0; i--) {
      num = num<<1;
      num += state[i];
    }
    return num;
  }

  inline size_t state2num(STATE const& state, BOND const& bond){
    size_t u = 0;
    for (int i=0; i<bond.size(); i++){
      // int tmp = cstate[bond[i]];
      u += (state[bond[i]] << i);
    }
    return u;
  }

  STATE num2state(int num, int L);
  std::string return_name(int dot_type, int op_type);


  static std::string op_type_name[2] = {
      "diagonal",
      "off-diagonal"
    };


  static std::string dot_type_name[3] = {
    "state",
    "operator",
    "worm"
  };


  template <size_t max_L = 4>
  std::array<size_t, max_L+1> pows_array(size_t sps = 2){
    std::array<size_t, max_L+1> arr; size_t x = 1;
    for (int i=0; i<max_L+1; i++){ 
      arr[i]=x; x*=sps;
    }
    return arr;
  }
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

/*
  the actual size of state (number of bits for expressing state ) is 2 * size
*/
template <size_t nls_, size_t max_L>
class spin_state::Operatorv2{
  const BOND* const bond_ptr_;
  // size_t s0_;
  // size_t s1_;
  static const size_t nls = nls_;
  static const size_t sps = (1<<nls);
  static const std::array<size_t, max_L+1> pows;
  size_t size_;
  size_t op_type_;
  size_t state_;
  double tau_;
public:
  Operatorv2() :bond_ptr_(nullptr){}
  // Operatorv2(){}


  //bond_, dot_labels_, size_, op_type, state_,tau_;
  Operatorv2(const BOND* const bp , size_t st,
            size_t si, size_t o, double t):bond_ptr_(bp), state_(st), size_(si), op_type_(o), tau_(t)
  {
    ASSERT(size_ == bp->size(), "bond size and size is inconsistent");
  }

  //size_, op_type, state_,tau_;
  Operatorv2(size_t st, size_t si, size_t o, double t)
  :state_(st), size_(si), op_type_(o), tau_(t), bond_ptr_(nullptr)
  {
    // ASSERT(size_ == bond_.size(), "bond size and size is inconsistent");
  }

  
  // void set_state(size_t sp) { state_ = sp; }
  size_t size() const {return size_;}
  size_t op_type()const {return op_type_;}
  size_t state()const {return state_;}
  size_t state(size_t dir)const { // dir = 0 lower part, dir = 1 upper pirt
    if (dir==0) return state_ % pows[size_];
    else if (dir == 1) return state_ / pows[size_];
    return -1;
  }
  double tau()const {return tau_;}
  int bond(int i) {return bond_ptr_->operator[](i);}
  const BOND* bond_ptr()const {return bond_ptr_;}
  // size_t s0() const {return s0_;}
  // size_t s1() const {return s1_;}

  /*
  leg = 0,1,2,3 for bond operator     
  2  3
  ====
  0  1.
  */

  void update_state(size_t leg, size_t fl=1)
    {
    size_t t = pows[leg+1];
    size_t a = pows[leg];
    state_ = (state_/t)*t + (state_%t+fl*a) % t;
    //  state_ ^= (fl << (nls*leg)); 
    }
  // SPIN get_spin(size_t leg) const {return (state_>>leg) & 1;}
  SPIN get_local_state(size_t leg) const {
    return (state_ % pows[leg]); 
  }
  bool is_off_diagonal() const{ return (state(0) != state(1)); }
  bool is_diagonal()const{ return !is_off_diagonal();}
  static Operatorv2 sentinel(double tau = 1){ return Operatorv2(0, 0, 0, tau);}
  void print(std::ostream& os) const {
    for (size_t i=0; i<size_*2; i++) os << get_local_state(i) << " ";
    os << tau_;
  }
  friend std::ostream& operator<<(std::ostream& os, Operatorv2 const& op) {
    op.print(os);
    return os;
  }
  STATE const get_state_vec(){
    STATE state_vec(size_*2);
    for (int i=0; i<size_*2; i++) state_vec[i] = get_local_state(i);
    return state_vec;
  }



  /*
  return label worm will move to
  params
  ------
  cindex : current index (0 to 3). corresponds to which leg the worm comes in.
  nindex : next index the worm goes out.
  clabel : label of dot. (label doesn't distinguish the direction worm goes out or comes in)
  L : number of site the operator acts, typically 2.

  */
  int next_dot(int cindex, int nindex, int clabel){
    // int cindex = GetIndex(ptr, 0);
    cindex %= size_;
    nindex %= size_;
    return clabel + (nindex - cindex);
  }
};


template <size_t nls_, size_t max_L>
const std::array<size_t, max_L+1> spin_state::Operatorv2<nls_, max_L>::pows = pows_array<max_L>(sps);