#pragma once

#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
#include "model.hpp"

using std::cout;
using std::endl;


namespace model{
  class BaseState;
  class BottomState;
  class OpState;
  class Worms;
  class Dot;
  using BaseStatePtr = std::shared_ptr<BaseState>;
  using OpStatePtr = std::shared_ptr<OpState>;
  using WormsPtr = std::shared_ptr<Worms>;
  using BStatePtr = std::shared_ptr<BottomState>;


  
}

class model::BaseState : public std::vector<int>
{
  typedef std::vector<int> vec;
  public :
  int L;
  int _size;
  const local_operator* plop = nullptr;

  BaseState(){}
  virtual std::ptrdiff_t GetIndex(int* ptr, int UorD = 0){
    return std::distance(this->data(), ptr);
  }

  virtual int* GetStatePtr(int* ptr,int UorD = 0){
    return ptr;
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
  BaseState(int L):vec(L, 0), L(L), _size(L){}

  BaseState(std::vector<int> state):vec(state), L(state.size()), _size(L){}

};

class model::BottomState : public BaseState
{
  public :
  double tau;
  BottomState(){}
  BottomState(int L, double t=0):BaseState(L), tau(0){}
  ~BottomState(){
  // cout << "Deconstructor (BottomState) was called" << endl;
}
};





class model::OpState : public BaseState
{
  public :
  const local_operator* plop;
  std::vector<int> bond;
  double tau;
  ~OpState(){
    // cout << "Deconstructor (OpState) was called" << endl;
  }
  OpState():plop(nullptr){}
  OpState(int L_, local_operator* plop, std::vector<int> bond, double t)
  :BaseState(2*L_), plop(plop), bond(bond), tau(t)
  {
    BaseState::L = L_;
    // ASSERT(l.size() == L_, "size of labels must be equal to given L");
    ASSERT(bond.size() == plop->L, "size of bond must be equal to operator size");
  }

  OpState(std::vector<int> state, local_operator* plop
          ,std::vector<int> bond, double t)
  :BaseState(state), plop(plop), bond(bond), tau(t)
  {
    BaseState::L = state.size()/2;
    // ASSERT(l.size() == L, "size of labels must be equal to given L");
    ASSERT(bond.size() == plop->L, "size of bond must be equal to operator size");
    ASSERT(L == plop->L, "in consistent error");
  }
  /*
  int UorD : 1 or 0, corresponds to upside or downside of state.
  */
  int* GetStatePtr (int* ptr, int UorD) override{
    return ptr + UorD*L;
  }

  std::ptrdiff_t GetIndex(int* ptr, int UorD) override{
    ptr = GetStatePtr(ptr, UorD);
    return std::distance(this->data(), ptr);
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

class model::Worms : public BaseState
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
class model::Dot
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
};
