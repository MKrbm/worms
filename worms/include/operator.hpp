#pragma once

#include <gtest/gtest-matchers.h>
#include <iostream>
#include <memory>
#include <iterator>
#include <tuple>
#include <vector>
#include "spin_state.hpp"
#include "funcs.hpp"

namespace spin_state
{
  class Operator
  {
    const VS *const bond_ptr_;
    const VS *const pows_ptr_;
    int op_type_; // type of bond operator (basically 1 or 2 types)
    size_t state_;
    state_t nn_state_;
    size_t cnt_;
    size_t sps;
    double tau_;

  public:
    Operator() : bond_ptr_(nullptr), pows_ptr_(nullptr) {}

    Operator(Operator const &op)
        : bond_ptr_(op.bond_ptr_), pows_ptr_(op.pows_ptr_), state_(op.state_), op_type_(op.op_type_), tau_(op.tau_), cnt_(0), sps(op.sps),
          nn_state_(op.nn_state_)
    {}

    bool same(Operator const &op){
      if (op_type_ != op.op_type_){
        std::cout << "op_type_ = " << op_type_ << " op.op_type_ = " << op.op_type_ << std::endl;
        std::cout << "state_ = " << state_ << " op.state_ = " << op.state_ << std::endl;
        std::cout << "tau_ = " << tau_ << " op.tau_ = " << op.tau_ << std::endl;
        std::cout << "nn_state_ = " << nn_state_ << " op.nn_state_ = " << op.nn_state_ << std::endl;
        return false;
      }
      if (state_ != op.state_){
        std::cout << "op_type_ = " << op_type_ << " op.op_type_ = " << op.op_type_ << std::endl;
        std::cout << "state_ = " << state_ << " op.state_ = " << op.state_ << std::endl;
        std::cout << "tau_ = " << tau_ << " op.tau_ = " << op.tau_ << std::endl;
        std::cout << "nn_state_ = " << nn_state_ << " op.nn_state_ = " << op.nn_state_ << std::endl;
        return false;
      }
      if (tau_ != op.tau_){
        std::cout << "op_type_ = " << op_type_ << " op.op_type_ = " << op.op_type_ << std::endl;
        std::cout << "state_ = " << state_ << " op.state_ = " << op.state_ << std::endl;
        std::cout << "tau_ = " << tau_ << " op.tau_ = " << op.tau_ << std::endl;
        std::cout << "nn_state_ = " << nn_state_ << " op.nn_state_ = " << op.nn_state_ << std::endl;
        return false;
      }
      if (nn_state_ != op.nn_state_){
        std::cout << "op_type_ = " << op_type_ << " op.op_type_ = " << op.op_type_ << std::endl;
        std::cout << "state_ = " << state_ << " op.state_ = " << op.state_ << std::endl;
        std::cout << "tau_ = " << tau_ << " op.tau_ = " << op.tau_ << std::endl;
        std::cout << "nn_state_ = " << nn_state_ << " op.nn_state_ = " << op.nn_state_ << std::endl;
        return false;
      }
      return true;
    }

    // Operator& operator == (Operator const &op)
    // {
    //   if (this == &op) return *this;
    // }

    Operator& operator=(Operator const &op)
    {
      state_ = op.state_;
      nn_state_ = op.nn_state_;
      op_type_ = op.op_type_;
      tau_ = op.tau_;
      cnt_ = 0;
      sps = op.sps;
      return *this;
    }

    Operator(const VS *const bp, const VS *pp, size_t st,
             int o, double t)
        : bond_ptr_(bp), pows_ptr_(pp), state_(st), op_type_(o), tau_(t), cnt_(0), sps(pp->at(1))
    {
      if (pp->size() != 2 * bp->size() + 1)
        throw std::invalid_argument("power size is not consistent with bond size");
    }

    Operator(size_t st, int o, double t)
        : state_(st), op_type_(o), tau_(t), bond_ptr_(nullptr), pows_ptr_(nullptr), cnt_(0)
    {
    }

    // d* if nn_state_ is given
    /*
    nn_state = [x0, x0_prime, x1, x2, .., x6]
    Only x0 can take off-diagonal values
    */
    Operator(const VS *const bp, const VS *pp, size_t st, state_t ss, int o, double t)
        : bond_ptr_(bp), op_type_(o), tau_(t), cnt_(0), nn_state_(ss), pows_ptr_(pp), sps(pp->at(1)), state_(st)
    {
      if (bond_ptr_->size() != nn_state_.size())
        throw std::invalid_argument("bond size is not consistent with single state size");
      if (pp->size() != 3)
        throw std::invalid_argument("power size should be 3 for single-site state");
      if (o >= 0)
        throw std::invalid_argument("op_type must be negative for single-site state");
    }

    size_t cnt() const { return cnt_; }
    void set_state(size_t s)
    {
      state_ = s;
    }
    void reset_cnt() { cnt_ = 0; }
    void add_cnt() { cnt_++; }
    size_t size() const { return bond_ptr_->size(); }
    int op_type() const { return op_type_; }
    size_t state() const
    {
      return state_;
    }
    size_t state(size_t dir) const
    { 
      //* if is_single() is true, size() is 1
      size_t _size = size(); 
      if (!_check_is_bond()) _size = 1;
      if (dir == 0)
        return state_ % (*pows_ptr_)[_size];
      else if (dir == 1)
        return state_ / (*pows_ptr_)[_size];
      else 
        throw std::invalid_argument("dir must be 0 or 1");
      return -1;
    }
    const state_t &nn_state() const
    {
      is_single();
      return nn_state_;
    }
    spin_t nn_state(size_t i) const
    {
      is_single();
      return nn_state_[i];
    }

    const state_t &update_nn_state(size_t index, int fl)
    {
      is_single();
      nn_state_[index] = (nn_state_[index] + fl) % sps;
      return nn_state_;
    }
    double tau() const { return tau_; }
    int bond(int i) { return bond_ptr_->operator[](i); }
    const VS *bond_ptr() const { return bond_ptr_; }
    const VS *pows_ptr() const { return pows_ptr_; }

    /*
    leg = 0,1,2,3 for bond operator
    2  3
    ====
    0  1.
    */
    size_t update_state(size_t leg, size_t fl)
    {
      size_t a = (*pows_ptr_)[leg];
      size_t t = (*pows_ptr_)[leg + 1];
      state_ = (state_ / t) * t + (state_ % t + fl * a) % t;
      return state_;
    }
    US get_local_state(size_t leg) const { return (state_ % (*pows_ptr_)[leg + 1]) / (*pows_ptr_)[leg]; }
    bool is_off_diagonal() const { return (state_ ? (state(0) != state(1)) : false); }
    bool is_diagonal() const { return !is_off_diagonal(); }
    static Operator sentinel(double tau = 1) { return Operator(0, 0, tau); }
    void print(std::ostream &os) const
    {
      for (size_t i = 0; i < size() * 2; i++)
        os << get_local_state(i) << " ";
      os << tau_;
    }
    friend std::ostream &operator<<(std::ostream &os, Operator const &op)
    {
      op.print(os);
      return os;
    }
    state_t get_state_vec() const
    {
      state_t state_vec(size() * 2);
      for (int i = 0; i < size() * 2; i++)
      {
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
    int next_dot(int cindex, int nindex, int clabel)
    {
      // int cindex = GetIndex(ptr, 0);
      cindex %= size();
      nindex %= size();
      return clabel + (nindex - cindex);
    }

    void is_bond() const
    {
      if (!_check_is_bond())
      {
        throw std::invalid_argument("Operator::is_bond(): invalid operator");
      }
    }

    void is_single() const
    {
      if (_check_is_bond()){
        std::cout << "op_type_ = " << op_type_ << " nn_state" << nn_state_ << std::endl;
        std::cout << "check_is_bond = " << _check_is_bond() << std::endl;
        throw std::invalid_argument("Operator::is_single(): invalid operator");
      }
    }

    /*
    return true if this operator is bond operator
    */
    bool _check_is_bond() const
    {
      if (nn_state_.size() == 0 && op_type_ >= 0)
      {
        return true;
      }
      else if (nn_state_.size() != 0 && op_type_ < 0)
      {
        return false;
      }
      else
      {
        throw std::invalid_argument("Operator::check_is_bond(): invalid operator");
      }
    }
  };
}
