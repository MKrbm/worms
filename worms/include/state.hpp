#pragma once

#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
#include "model.hpp"

using std::cout;
using std::endl;


namespace spin_state{
  class BaseState;
  class BottomState;
  class OpState;
  class Worms;
  class Dot;
  class Dotv2;
  class BottomStatev2;
  class OpStatev2;
  class Wormsv2;
  class Operatorv2;
  

  using STATE = model::STATE;
  using BaseStatePtr = std::shared_ptr<BaseState>;
  using OpStatePtr = std::shared_ptr<OpState>;
  using WormsPtr = std::shared_ptr<Worms>;
  using BStatePtr = std::shared_ptr<BottomState>;
  using local_operator = model::local_operator;
  using WORM_ARR = std::vector<std::tuple<int, int, int, double>>; //  site, spin, dot_label, tau (dot label is needed for reverse lookup)
  using WORM = std::tuple<int, int, int, double>;
  using DOT_ARR = std::vector<std::tuple<int,int,int,int>>;   //prev, next, dot_type, index, (index refers to the legs of the dot with respect to the class of dots)

  /*

  this function works even if L < state.size().
  In this case, only consider state[:L]

  params
  -----
  int[] state : vector of 1 or -1. 
  int L : size of state

  return
  ------
  integer representation of state

  */
  inline int state2num(std::vector<int> const& state, int L = -1){
    int num = 0;
    if (L < 0) L = state.size();
    for (int i = L-1; i >= 0; i--) {
      num = num<<1;
      num += state[i];
    }
    return num;
  }

  inline int state2num(std::vector<int> const& state, std::vector<int> const& bond){
    int u = 0;
    for (int i = bond.size()-1; i >= 0; i--) {
      u <<= 1;
      u |= state[bond[i]];
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


  template<unsigned int NUM_LEGS, unsigned int DIM> struct spin_state;

  template<>
  struct spin_state<2, 2> {
    static const int num_configurations = 16;
    static const int num_candidates = 4;
    static int p2c(int p, int l) { return (p >> l) & 1; }
    static int p2u(int p, int d) { return (p >> (2 * d)) & 3; }
    static int c2u(int c0, int c1) { return (c0 | (c1 << 1)); }
    static int c2p(int c0, int c1, int c2, int c3) {
      return (c0 | (c1 << 1) | (c2 << 2) | (c3 << 3));
    }
    static int u2p(int u0, int u1) { return (u0 | (u1 << 2)); }
    static int candidate(int p, int g) { return p ^ (1<<g); }
    static int maskp(int l) { return (1 << l); }
    static bool is_diagonal(int p) { return p2u(p, 0) == p2u(p, 1); }
  };
}

class spin_state::BaseState : public std::vector<int>
{
  typedef std::vector<int> vec;
  public :
  int L;
  int _size;
  const double tau;
  const std::vector<int> bond;
  local_operator* const plop;

  BaseState() : tau(0), plop(nullptr){}

  BaseState(int L, double tau = 0, local_operator* ptr = nullptr, std::vector<int> bond = std::vector<int>())
  :vec(L, 0), L(L), _size(L), plop(ptr), bond(bond), tau(tau) {}

  BaseState(std::vector<int> state, double tau = 0, local_operator* ptr = nullptr, std::vector<int> bond = std::vector<int>())
  :vec(state), L(state.size()), _size(L), plop(ptr), bond(bond), tau(tau) {}


  virtual std::ptrdiff_t GetIndex(int* ptr, int dir_in = 0){
    return std::distance(this->data(), ptr);
  }

  virtual int* GetStatePtr(int* ptr,int dir_in = 0){
    return ptr;
  }

  virtual int GetLabel(int cindex, int nindex, int clabel){
    // int cindex = GetIndex(ptr, 0);
    std::cerr << "GetLabel is unavailable" << std::endl;
    return 0;
  }

  virtual int GetNum(){
    return state2num(*this, _size);
  }

  virtual bool is_off_diagonal(){
    return true;
  }
  virtual bool is_diagonal(){
    return true;
  }
  virtual ~BaseState(){
    // cout << "Deconstructor was called" << endl;
  }

  void push_back (int x){
    ASSERT(false, "push_back is unavailable"); //inorder to avoid pointer problem.
  }

  void resize (int x){
    ASSERT(false, "resize is unavailable");
  }

  int get_size(){
    return _size;
  }

};

class spin_state::BottomState : public BaseState
{
  public :
  BottomState(){}
  BottomState(int L, double t=0):BaseState(L, t){}
  ~BottomState(){
  // cout << "Deconstructor (BottomState) was called" << endl;
}
};





class spin_state::OpState : public BaseState
{
  public :

  ~OpState(){
    // cout << "Deconstructor (OpState) was called" << endl;
  }
  OpState(){}

  OpState(int L_, local_operator* plop, std::vector<int> bond_, double t)
  :BaseState(2*L_, t, plop, bond_)
  {
    BaseState::L = L_;
    // ASSERT(l.size() == L_, "size of labels must be equal to given L");
    ASSERT(bond.size() == plop->L, "size of bond must be equal to operator size");
  }

  OpState(std::vector<int> state, local_operator* plop
          ,std::vector<int> bond_, double t)
  :BaseState(state, t, plop, bond_)
  {
    BaseState::L = state.size()/2;
    // ASSERT(l.size() == L, "size of labels must be equal to given L");
    ASSERT(bond.size() == plop->L, "size of bond must be equal to operator size");
    ASSERT(L == plop->L, "in consistent error");
  }

  OpState(double t): BaseState(0, t){} //* for append sentinels.
  /*
  int dir : 1 or 0, corresponds to upside or downside of state.
  */
  int* GetStatePtr (int* ptr, int dir_in) override{
    return ptr + dir_in*L;
  }

  /*
  return index of element
  params
  ------
  int* ptr : ptr to element of state
  int dir_in : direction (1 or 0) worm comes in.
  */
  std::ptrdiff_t GetIndex(int* ptr, int dir_in) override{
    ptr = GetStatePtr(ptr, dir_in);
    return std::distance(this->data(), ptr);
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
  int GetLabel(int cindex, int nindex, int clabel) override{
    // int cindex = GetIndex(ptr, 0);
    cindex %= L;
    nindex %= L;
    return clabel + (nindex - cindex);
  }

  bool is_off_diagonal() override{
    for (int i = 0; i<L; i++){
      if ((*this)[i] != (*this)[i+L]) return true;
    }
    return false;
  }

  bool is_diagonal() override{
    return !is_off_diagonal();
  }

  
};

class spin_state::Worms : public BaseState
{
  public :
  const std::vector<int>& spin;
  const std::vector<int>& site;
  // std::vector<int> worm_site;
  ~Worms(){
    // cout << "Deconstructor (Worms) was called" << endl;
  }
  Worms():BaseState(), spin(*this), site(bond){}
  Worms(int L)
  :BaseState(L), spin(*this), site(bond){}

  Worms(std::vector<int> spin_
          ,std::vector<int> site_, double t)
  :BaseState(spin_, t, nullptr, site_), spin(*this), site(bond)
  {
    BaseState::L = 1;
    // ASSERT(l.size() == L, "size of labels must be equal to given L");
    ASSERT(bond.size() == 1, "size of site (called bond here) must be equal to 1");
    ASSERT(size() == 1, "the size ofspin (state) must be equal to 1");
  }

  Worms(int spin_
          ,int site_, double t)
  :BaseState(std::vector<int>(1,spin_), t, nullptr, std::vector<int>(1,site_)), spin(*this), site(bond)
  {
    BaseState::L = 1;
    // ASSERT(l.size() == L, "size of labels must be equal to given L");
    ASSERT(bond.size() == 1, "size of site (called bond here) must be equal to 1");
    ASSERT(size() == 1, "the size ofspin (state) must be equal to 1");
  }
};



/*
params
------
int site : site the dot is at
int tau : tau
int prev : previous dot label
int next : next dot label
int% sptr : ptr to state
BaseStatePtr typeptr : ptr to state type class
int dot_type : state type 0 : bottom state, 1:operator, 2:worms;

*/
class spin_state::Dot
{
  private:
  
  public:
  int prev;
  int next = -1;
  int site;
  int dot_type;
  int* sptr;
  double tau;
  BaseStatePtr typeptr;
  Dot(int s, int p, int n, int* sptr, BaseState* type, int d)
  :site(s),  prev(p), next(n), sptr(sptr), typeptr(BaseStatePtr(type)), dot_type(d)
  {}

  Dot(int s, int p, int n, int* sptr, BaseStatePtr type, int d)
  :site(s),  prev(p), next(n), sptr(sptr), typeptr(type), dot_type(d)
  {}

  Dot(){}
  
  void set_next(int n){
    next = n;
  }
  void set_prev(int p){
    prev = p;
  }

  // static add_origin(int s, )

  /*
  int dir : direction worm goes
  */
  int move_next(int dir){
    // if (dir == 1) return next;
    // else if (dir == 0) return prev;
    return (dir == 0) ? prev : next;
    ASSERT(false, "dir can be 1 or 0");
  }

};

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
  int prev_;
  int next_;
  int dot_type_;
  int index_; 
  int site_;
public:
  Dotv2(){}
  Dotv2(int s, int p, int n, int o, int i)
  :site_(s), prev_(p), next_(n), dot_type_(o), index_(i)
  {}

  static Dotv2 state(int s) { return Dotv2(s, s, s, -1, s); }
  static Dotv2 worm(int s, int p, int n, int wl) { return Dotv2(s, p, n, -2, wl); }
  int move_next(int dir){
    return (dir == 0) ? prev_ : next_;
    ASSERT(false, "dir can be 1 or 0");
  }
  int site() const { return site_; }
  int prev() const { return prev_; }
  int next() const { return next_; }
  int leg(int dir, int L) const {
    if (at_operator()) return dir*L + index_;
    else return 0;
  }
  int label() const {
    if (at_operator()) return dot_type_;
    else return index_;
  }
  bool at_operator() const { return dot_type_ >= 0; }
  bool at_origin() const { return dot_type_ == -1; }
  bool at_worm() const { return dot_type_ == -2; }
  void set_prev(int p) { prev_ = p; }
  void set_next(int n) { next_ = n; }
  int move_next(int dir) const {
    // if (dir == 1) return next;
    // else if (dir == 0) return prev;
    ASSERT(false, "dir can be 1 or 0");
    return (dir == 0) ? prev_ : next_;
  }
};


class spin_state::Wormsv2{
  int site_;
  int spin_;
  int dot_label_;
  double tau_;
public:
  Wormsv2(){}
  Wormsv2(int si, int sp, int dl, double t):site_(si), spin_(sp),dot_label_(dl),tau_(t)
  {}
  
  void set_spin(int s) { spin_ = s; }
  int site() const {return site_;}
  int spin()const {return spin_;}
  int dot_label()const {return dot_label_;}
  double tau()const {return tau_;}
};

/*

  the actual size of state (number of bits for expressing state ) is 2 * size
*/
class spin_state::Operatorv2{
  int s0_, s1_;
  int size_;
  int op_type_;
  int state_;
  double tau_;
public:
  Operatorv2(){}

  //bond_, dot_labels_, size_, op_type, state_,tau_;
  Operatorv2(int s0, int s1 , int st,
            int si, int o, double t):s0_(s0), s1_(s1), state_(st), size_(si), op_type_(o), tau_(t)
  {
    ASSERT(size_ == b.size(), "bond size and size is inconsistent");
  }

  //size_, op_type, state_,tau_;
  Operatorv2(int st, int si, int o, double t)
  :state_(st), size_(si), op_type_(o), tau_(t)
  {
    ASSERT(size_ == bond_.size(), "bond size and size is inconsistent");
  }

  
  void set_state(int sp) { state_ = sp; }
  int size() const {return size_;}
  int op_type()const {return op_type_;}
  int state()const {return state_;}
  int state(int dir)const { // dir = 0 lower part, dir = 1 upper pirt
    if (dir==0) return state_ & ((1<<size_)-1);
    else if (dir == 1) return (state_ >> size_) & ((1<<size_)-1);
    return -1;
  }
  double tau()const {return tau_;}
  int s0() const {return s0_;}
  int s1() const {return s1_;}

  // int bond(int s) const {return bond_[s];}
  // std::vector<int> const & bond() const {return bond_;}
  // int dot_labels(int s) const {return dot_labels_[s];}
  // std::vector<int> const & dot_labels() const {return dot_labels_;}

  /*
  leg = 0,1,2,3 for bond operator     
  2  3
  ====
  0  1.
  */
  void flip_state(int leg){ state_ ^= (1<<leg);} 

  int get_spin(int leg) const {return (state_>>leg) & 1;}

  bool is_off_diagonal() const{
    if (state(0) != state(1)) return true;
    return false;
  }
  bool is_diagonal()const{
    return !is_off_diagonal();
  }

  void print(std::ostream& os) const {
    for (int i=0; i<size_*2; i++) os << get_spin(i) << " ";
    os << tau_;
  }

  std::vector<int> const get_state_vec(){
    std::vector<int> state_vec(size_*2);
    for (int i=0; i<size_*2; i++) state_vec[i] = get_spin(i);
    return state_vec;
  }

  static Operatorv2 sentinel(double tau = 1){
    return Operatorv2(0, 0, 0, tau);
  }

  friend std::ostream& operator<<(std::ostream& os, Operatorv2 const& op) {
    op.print(os);
    return os;
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