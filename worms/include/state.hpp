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
  int state2num(STATE_ const& state, int L){
    int num = 0;
    if (L < 0) L = state.size();
    for (int i = L-1; i >= 0; i--) {
      num = num<<1;
      num += state[i];
    }
    return num;
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
  const std::vector<int> bond;
  const local_operator* plop = nullptr;

  BaseState(){}
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
    ASSERT(false, "push_back is unavailable");
  }

  void resize (int x){
    ASSERT(false, "resize is unavailable");
  }

  int get_size(){
    return _size;
  }



  BaseState(int L):vec(L, 0), L(L), _size(L){}

  BaseState(std::vector<int> state):vec(state), L(state.size()), _size(L){}

};

class spin_state::BottomState : public BaseState
{
  public :
  double tau;
  BottomState(){}
  BottomState(int L, double t=0):BaseState(L), tau(0){}
  ~BottomState(){
  // cout << "Deconstructor (BottomState) was called" << endl;
}
};





class spin_state::OpState : public BaseState
{
  public :
  const local_operator* plop;
  const std::vector<int> bond;
  double tau;
  ~OpState(){
    // cout << "Deconstructor (OpState) was called" << endl;
  }
  OpState():plop(nullptr){}
  OpState(int L_, local_operator* plop, std::vector<int> bond_, double t)
  :BaseState(2*L_), plop(plop), bond(bond_), tau(t)
  {
    BaseState::L = L_;
    // ASSERT(l.size() == L_, "size of labels must be equal to given L");
    ASSERT(bond.size() == plop->L, "size of bond must be equal to operator size");
  }

  OpState(std::vector<int> state, local_operator* plop
          ,std::vector<int> bond_, double t)
  :BaseState(state), plop(plop), bond(bond_), tau(t)
  {
    BaseState::L = state.size()/2;
    // ASSERT(l.size() == L, "size of labels must be equal to given L");
    ASSERT(bond.size() == plop->L, "size of bond must be equal to operator size");
    ASSERT(L == plop->L, "in consistent error");
  }
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
  std::vector<double> tau_list;
  std::vector<int> worm_site;
  ~Worms(){
    // cout << "Deconstructor (Worms) was called" << endl;
  }
  Worms(){}
  Worms(int L)
  :BaseState(L), tau_list(std::vector<double>(L)), worm_site(std::vector<int>(L)){}
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
  public:
  int prev;
  int next = -1;
  int site;
  int dot_type;
  int* sptr;
  double tau;
  BaseStatePtr typeptr;
  Dot(int s, double t , int p, int n, int* sptr, BaseState* type, int d)
  :site(s), tau(t), prev(p), next(n), sptr(sptr), typeptr(BaseStatePtr(type)), dot_type(d)
  {}

  Dot(int s, double t , int p, int n, int* sptr, BaseStatePtr type, int d)
  :site(s), tau(t), prev(p), next(n), sptr(sptr), typeptr(type), dot_type(d)
  {}

  Dot(){}
  
  void set_next(int n){
    next = n;
  }
  void set_prev(int p){
    prev = p;
  }

  /*
  int dir : direction worm goes
  */
  int move_next(int dir){
    if (dir == 1) return next;
    else if (dir == 0) return prev;
    ASSERT(false, "dir can be 1 or 0");
  }

};
