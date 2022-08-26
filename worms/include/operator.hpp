#pragma once

#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
#include <vector>
#include "state2.hpp"


namespace spin_state{

  class Operator{
    const VS* const bond_ptr_;
    const VS* const pows_ptr_;
    size_t op_type_;
    size_t state_;
    size_t cnt_;
    double tau_;
  public:
    Operator() :bond_ptr_(nullptr), pows_ptr_(nullptr){}
    
    Operator(const VS* const bp, const VS* pp, size_t st,
            size_t o, double t)
            :bond_ptr_(bp), pows_ptr_(pp), state_(st), op_type_(o), tau_(t), cnt_(0)
    {}

    Operator(size_t st, size_t o, double t)
    :state_(st), op_type_(o), tau_(t), bond_ptr_(nullptr), pows_ptr_(nullptr), cnt_(0)
    {}

    size_t cnt() const {return cnt_;}
    void set_state(size_t s) { state_ = s; cnt_ = 0;}
    void add_cnt() {cnt_++;}
    size_t size() const {return bond_ptr_->size();}
    size_t op_type()const {return op_type_;}
    size_t state()const {return state_;}
    size_t state(size_t dir)const { // dir = 0 lower part, dir = 1 upper pirt
      if (dir==0) return state_ % (*pows_ptr_)[size()];
      else if (dir == 1) return state_ / (*pows_ptr_)[size()];
      return -1;
    }
    double tau()const {return tau_;}
    int bond(int i) {return bond_ptr_->operator[](i);}
    const VS* bond_ptr()const {return bond_ptr_;}
    const VS* pows_ptr()const {return pows_ptr_;}

    /*
    leg = 0,1,2,3 for bond operator     
    2  3
    ====
    0  1.
    */
    void update_state(size_t leg, size_t fl)
      {
      size_t a = (*pows_ptr_)[leg];
      size_t t = (*pows_ptr_)[leg+1];
      state_ = (state_/t)*t + (state_%t+fl*a) % t;
      }
    US get_local_state(size_t leg) const { return (state_%(*pows_ptr_)[leg+1])/(*pows_ptr_)[leg];}
    bool is_off_diagonal() const{ return (state(0) != state(1));}
    bool is_diagonal()const{ return !is_off_diagonal();}
    static Operator sentinel(double tau = 1){ return Operator(0, 0, tau);}
    void print(std::ostream& os) const {
      for (size_t i=0; i<size()*2; i++) os << get_local_state(i) << " ";
      os << tau_;
    }
    friend std::ostream& operator<<(std::ostream& os, Operator const& op) {
      op.print(os);
      return os;
    }
    VUS const get_state_vec(){
      VUS state_vec(size()*2);
      for (int i=0; i<size()*2; i++) {
        state_vec[i] = get_local_state(i);
      }
      return state_vec;
    }


    /*
    return label worm move to
    params
    ------
    cindex : current index (0 to 3). corresponds to which leg the worm comes in.
    nindex : next index the worm goes out.
    clabel : label of dot. (label doesn't distinguish the direction worm goes out or comes in)
    L : number of site the operator acts, typically 2.
    */
    int next_dot(int cindex, int nindex, int clabel){
      // int cindex = GetIndex(ptr, 0);
      cindex %= size();
      nindex %= size();
      return clabel + (nindex - cindex);
    }
  };

  // STATE num2state(int num, int L);
  // std::string return_name(int dot_type, int op_type);


  // template <size_t max_L = 4>
  // std::array<size_t, max_L+1> pows_array(size_t sps = 2){
  //   std::array<size_t, max_L+1> arr; size_t x = 1;
  //   for (int i=0; i<max_L+1; i++){ 
  //     arr[i]=x; x*=sps;
  //   }
  //   return arr;
  // }
}