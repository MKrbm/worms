#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <random>
#include <tuple>
#include <memory>
#include <fstream>
#include <unordered_map>

#include <bcl.hpp>
#include <alps/alea/core.hpp>
#include <alps/alea/computed.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/internal/galois.hpp>
#include <alps/alea/var_strategy.hpp>
#include <alps/alea/batch.hpp>
#include <alps/alea/autocorr.hpp>


#include "load_npy.hpp"
#include "automodel.hpp"

// batch_obs type is used to store results of observables
typedef alps::alea::batch_acc<double> batch_obs;
typedef alps::alea::batch_result<double> batch_res;

namespace model
{
  class observable;
  class BaseWormObs;
  class ArrWormObs;
  class WormObs;
  class MapWormObs;
  using worm_pair_t = std::pair<std::string, model::WormObs>;
}

/*
class that contains observable
We assume that observable $\hat{O}$ are given by sum of local operators $\hat{O} = \sum_{i} \sum_t \hat{O^t}_i $ where t is type of operator and i runs all sites (or bonds).
Usually this operator type t corresponds to type of local hamiltonian, and this class can only handle such case.

params
------
vector<vector<double>> obs_operators:
  vector of vector of double.
  obs_operators[t][s] is the element of local operator of type t at state s.
*/
class model::observable
{
private:
  std::vector<std::vector<double>> _obs_operators;
  int x;

public:
  double obs_operators(int t, int s) const { return _obs_operators[t][s]; }
  template <class MC>
  observable(base_model<MC> &spin_model, std::string obs_path, bool print = false)
  {
    using MCT = MC;
    const std::vector<local_operator<MCT>> loperators = spin_model.get_local_operators();
    int N_bond_t = loperators.size(); // number of bond types.u

    // * load path list
    std::vector<std::string> path_list;
    if (obs_path != "")
    {
      get_npy_path(obs_path, path_list);
      if (N_bond_t != path_list.size())
      {
        std::cerr << "number of types observable does not match to number of local operators (or # of bond types)" << std::endl;
        exit(1);
      }

      std::sort(path_list.begin(), path_list.end());

      for (int p_i = 0; p_i < N_bond_t; p_i++)
      {
        model::local_operator<MCT> loperator = loperators[p_i];
        std::string path = path_list[p_i];
        auto pair = load_npy(path);
        VS shape = pair.first;
        VD data = pair.second;
        size_t S = shape[0];
        if (shape[0] != shape[1])
        {
          std::cerr << "require square matrix" << std::endl;
          exit(1);
        }
        if (data.size() != loperator.ham_prime.size() * loperator.ham_prime.size())
        {
          std::cerr << "size of observable does not match to size of local operator" << std::endl;
          exit(1);
        }
        if (print)
          std::cout << "obs operator is read from " << path << std::endl;
        std::vector<double> _obs(S * S);
        for (int i = 0; i < S; i++)
          for (int j = 0; j < S; j++)
          {
            double y = data[i * S + j];
            double x = loperator.ham_vector(i * S + j); // x is absolute of the $\hat{h}_\prime[j][i]$ (note that $\hat{h}_\prime$ is a local hamiltonian whose origin shifts by ene_shift from original local hamiltonian)
            if (abs(y) < 1e-8)
            {
              _obs[i * S + j] = 0;
              continue;
            }
            if (abs(x) < 1e-8)
              x = 0;
            if (x == 0)
            {
              std::cerr << "denominator is zero for " << i << " " << j << "which lead to infinite variance";
              exit(1);
            }
            _obs[i * S + j] = y / x;
          }
        _obs_operators.push_back(_obs);
      }
    }
    else
    {
      for (int p_i = 0; p_i < N_bond_t; p_i++)
      {
        model::local_operator<MCT> loperator = loperators[p_i];
        std::vector<double> _obs(loperator.ham_prime.size() * loperator.ham_prime.size(), 0);
        _obs_operators.push_back(_obs);
      }
    }
  }
};

class model::BaseWormObs
{
protected:
  size_t _spin_dof; // spin degree of freedom.
  size_t _leg_size; // leg size of BaseWormObs either 1 or 2. 1 for one-site operator and 2 for bond operators.
  bool _has_operator = false;

public:
  BaseWormObs(size_t spin_dof, size_t legs);
  size_t spin_dof() const { return _spin_dof; }
  size_t leg_size() const { return _leg_size; }
  bool is_one_site() const { return _leg_size == 1; }
  bool has_operator() const { return _has_operator; }
  bool check_is_symm() const;
  bool check_no_single() const;   // check if there is no element that flip only single spin.
  bool check_no_diagonal() const; // check if there is no diagonal elements

  /*
  return local states of worm observable.

  t   h
  ↓   ↓
  2   3    ← x'
  ----
  0   1    ← x

  If this is one site operator, this only checks tail states.

  params
  ------
  spins = {x_t, x_h, x_t_p, x_h_p}
  x_h : spin state at lower portion of worm head.
  x_h_p : spin state at upper portion of worm head.
  x_t : spin state at lower portion of worm tail.
  x_t_p : spin state at upper portion of worm tail.
  */
  int GetState(size_t x1,
               size_t x2,
               size_t x3,
               size_t x4) const;
  int GetState(size_t x1, size_t x2) const;

  /*
  params
  ------
  r : distance between head and tail.
  tau : time difference between head and tail.
  */
  double operator()(std::array<size_t, 4> spins, double r, double tau) const;
  // double operator() (std::array<size_t, 4> spins) const;
  // double operator() (std::array<size_t, 2> spins) const;

  double operator()(size_t x1, size_t x2, size_t x3, size_t x4) const;
  double operator()(size_t x1, size_t x2) const;

  virtual double _operator(size_t state) const;
};

// Worm Obs where operator is calculated from matrix elements.
class model::ArrWormObs : public BaseWormObs
{
private:
  std::vector<double> _worm_obs;

public:
  ArrWormObs(size_t sdof, size_t legs);
  void _SetWormObs(std::vector<double> _obs, bool print = true);
  void ReadNpy(std::string path, bool print=false);
  double _operator(size_t state) const;
  std::vector<double> worm_obs() const { return _worm_obs; }
};

class model::WormObs : public batch_obs
{
public:
  typedef std::shared_ptr<BaseWormObs> BaseWormObsPtr;

private:
  BaseWormObsPtr _first, _second;
  size_t _spin_dof;

public:
  WormObs(){};
  WormObs(size_t spin_dof, std::string folder_path = "", bool print = false);
  WormObs(size_t spin_dof, std::pair<BaseWormObsPtr, BaseWormObsPtr> obspt_pair); // obs1 : one site worm operator, obs2 : bond worm operator.
  void add(std::array<size_t, 4> spins, size_t L, int sign, double r, double tau);
  static std::pair<BaseWormObsPtr, BaseWormObsPtr> ReadNpy(size_t spin_dof, std::string folder_path, bool print = false);
  const BaseWormObsPtr first() const { return _first; }
  const BaseWormObsPtr second() const { return _second; }
  // std::pair<BaseWormObsPtr, BaseWormObsPtr> worm_obs_ptr() const {return _worm_obs_ptr;}

  batch_obs &get_batch_obs() { return *this; }
  WormObs &operator<<(double x)
  {
    static_cast<batch_obs &>(*this) << x;
    return *this;
  }
};


class model::MapWormObs
{
private:
  std::unordered_map<std::string, WormObs> _worm_obs_map;
public:
  std::unordered_map<std::string, WormObs>& operator()(){return _worm_obs_map;}
  WormObs& operator[](std::string key) { 
    if (_worm_obs_map.count(key) == 0){throw std::runtime_error("Given key doesn't exist");}
    return _worm_obs_map[key]; 
  }
  // WormObs& operator[](std::string key) { return _worm_obs_map[key]; }

  MapWormObs(WormObs obs){
    _worm_obs_map["G"] = obs;
  }

  void push_back(std::string key, WormObs obs){
    _worm_obs_map[key] = obs;
  }

  //n* recursive template approach
  template<typename... Args>
  MapWormObs(worm_pair_t arg, Args... args );
  MapWormObs(worm_pair_t arg);
  //n* if char* is given, convert it to string.
  template<typename Arg, typename... Args>
  MapWormObs(Arg arg, Args... args )
  :MapWormObs(args...)
  {
    _worm_obs_map[arg.first] = arg.second;
  }

  template<typename Arg>
  MapWormObs(Arg arg)
  {
    cout << "Warning! Encountered Implicit type conversion. pair<string, T> to pair<const char*, T>" << endl;
    _worm_obs_map[arg.first] = arg.second;
  }
  MapWormObs() {}
};


extern template model::MapWormObs::MapWormObs(worm_pair_t, worm_pair_t);
extern template model::MapWormObs::MapWormObs(worm_pair_t, worm_pair_t, worm_pair_t);
extern template model::MapWormObs::MapWormObs(worm_pair_t, worm_pair_t,worm_pair_t, worm_pair_t);

// template model::MapWormObs::MapWormObs<worm_pair_t>(worm_pair_t);

