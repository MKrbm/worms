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

  using STATE = model::STATE;
  using BaseStatePtr = std::shared_ptr<BaseState>;
  using OpStatePtr = std::shared_ptr<OpState>;
  using WormsPtr = std::shared_ptr<Worms>;
  using BStatePtr = std::shared_ptr<BottomState>;
  using local_operator = model::local_operator;

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
  template<typename STATE_>
  int state2num(STATE_ const& state, int L = -1){
    int num = 0;
    if (L < 0) L = state.size();
    for (int i = L-1; i >= 0; i--) {
      num = num<<1;
      num += state[i];
    }
    return num;
  }

  template<typename STATE_>
  int state2num(STATE_ const& state, std::vector<int> const& bond){
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
int dot_type : state type where -1 : state, -2 : worms, non-negative integer : operator. 
int index : index which will be used to indexing the spin. e.g. if dot_type = -1. state[index] is the spin on the dot, if -2, worm[index], and so on.

*/
class spin_state::Dotv2
{
  int prev_;
  int next_;
  int dot_type_;
  int index_; 
public:
  Dotv2(){}
  Dotv2(int p, int n, int d, int i)
  :prev_(p), next_(n), index_(i), dot_type_(d)
  {}

  static Dotv2 state(int s) { return Dotv2(s, s, -1, 0); }
  static Dotv2 worm(int p, int n) { return Dotv2(p, n, -2, 0); }
  int move_next(int dir){
    return (dir == 0) ? prev_ : next_;
    ASSERT(false, "dir can be 1 or 0");
  }
  int prev() const { return prev_; }
  int next() const { return next_; }
  int index() const { return index_; }
  int op_label() const { return dot_type_; }
  bool at_operator() const { return dot_type_ >= 0; }
  bool at_origin() const { return dot_type_ == -1; }
  bool at_worm() const { return dot_type_ == -2; }
  void set_prev(int p) { prev_ = p; }
  void set_next(int n) { next_ = n; }
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
  
  void set_spin(int s){ spin_ = s; }
  int site(){return site_;}
  int spin(){return spin_;}
  int dot_label(){return dot_label_;}
  double tau(){return tau_;}

};
