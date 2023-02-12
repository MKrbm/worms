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

#include <bcl.hpp>
#include <alps/alea/core.hpp>
#include <alps/alea/computed.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/internal/galois.hpp>
#include <alps/alea/var_strategy.hpp>
#include <alps/alea/batch.hpp>

#include "load_npy.hpp"
#include "automodel.hpp"


// batch_obs type is used to store results of observables
typedef alps::alea::batch_acc<double> batch_obs;
typedef alps::alea::batch_result<double> batch_res;


namespace model{
  class observable;
  class BaseWormObs;
  class ArrWormObs;
  class WormObs;
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
    double obs_operators(int t, int s) const {return _obs_operators[t][s];}
    template <class MC>
    observable(base_model<MC> &spin_model, std::string obs_path, bool print=false)
    {
      using MCT = MC;
      const std::vector<local_operator<MCT>> loperators = spin_model.get_local_operators();
      int N_bond_t = loperators.size(); // number of bond types.u

      // * load path list
      std::vector<std::string> path_list;
      if (obs_path != ""){
        get_npy_path(obs_path, path_list);
        if (N_bond_t != path_list.size()){
          std::cerr << "number of types observable does not match to number of local operators (or # of bond types)" << std::endl;
          exit(1);
        }

        std::sort(path_list.begin(), path_list.end());

        for (int p_i=0; p_i<N_bond_t; p_i++) {
          model::local_operator<MCT> loperator = loperators[p_i];
          std::string path = path_list[p_i];
          auto pair = load_npy(path);
          VS shape = pair.first;
          VD data = pair.second;
          size_t S = shape[0];
          if (shape[0]!= shape[1]){ std::cerr << "require square matrix" << std::endl; exit(1); }
          if (data.size() != loperator.ham_prime.size() * loperator.ham_prime.size()){ std::cerr << "size of observable does not match to size of local operator" << std::endl; exit(1); }
          if (print) std::cout << "obs operator is read from " << path << std::endl;
          std::vector<double> _obs(S*S);
          for (int i=0; i<S; i++) for (int j=0; j<S; j++)
          {
            double y = data[i * S + j];
            double x = loperator.ham_vector(i * S + j); // x is absolute of the $\hat{h}_\prime[j][i]$ (note that $\hat{h}_\prime$ is a local hamiltonian whose origin shifts by ene_shift from original local hamiltonian)
            if (abs(y) < 1e-8) {_obs[i * S + j] = 0; continue;}
            if (abs(x) < 1e-8) x = 0;
            if (x == 0) { std::cerr << "denominator is zero for " << i << " " << j << "which lead to infinite variance"; exit(1); }
            _obs[i * S + j] = y / x;
          }
          _obs_operators.push_back(_obs);
        }
      } else {
        for (int p_i=0; p_i<N_bond_t; p_i++) {
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
    size_t spin_dof() const {return _spin_dof;}
    size_t leg_size() const {return _leg_size;}
    bool is_one_site() const {return _leg_size == 1;}
    bool has_operator() const {return _has_operator;}
    bool check_is_symm() const;
    bool check_no_single() const; // check if there is no element that flip only single spin.
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
    int GetState(std::array<size_t, 4> spins) const;
    int GetState(std::array<size_t, 2> spins) const;


    /*
    params
    ------  
    r : distance between head and tail.
    tau : time difference between head and tail.
    */
    double operator() (std::array<size_t, 4> spins, double r, double tau) const;
    double operator() (std::array<size_t, 4> spins) const;
    double operator() (std::array<size_t, 2> spins) const;


    virtual double _operator(size_t state) const ;
};


//Worm Obs where operator is calculated from matrix elements.
class model::ArrWormObs : public BaseWormObs
{
  private: 
    std::vector<double> _worm_obs;
  public:
    ArrWormObs(size_t sdof, size_t legs);
    void _SetWormObs(std::vector<double> _obs);
    void ReadNpy(std::string path);
    double _operator(size_t state) const;
    std::vector<double> worm_obs() const {return _worm_obs;}
};

class model::WormObs : public batch_obs
{
  public:
    typedef std::shared_ptr<BaseWormObs> BaseWormObsPtr;
  private:
    BaseWormObsPtr _first, _second;
    size_t _spin_dof;
  public:
    WormObs(size_t spin_dof, std::string folder_path = "", bool print = false);
    WormObs(size_t spin_dof, std::pair<BaseWormObsPtr, BaseWormObsPtr> obspt_pair); // obs1 : one site worm operator, obs2 : bond worm operator.
    void add(std::array<size_t, 4> spins, size_t L, int sign, double r, double tau);
    static std::pair<BaseWormObsPtr, BaseWormObsPtr> ReadNpy(size_t spin_dof, std::string folder_path, bool print = false);
    const BaseWormObsPtr first() const {return _first;}
    const BaseWormObsPtr second() const {return _second;}
    // std::pair<BaseWormObsPtr, BaseWormObsPtr> worm_obs_ptr() const {return _worm_obs_ptr;}

    batch_obs& get_batch_obs() {return *this;}
    WormObs& operator<< (double x) {static_cast<batch_obs&>(*this) << x; return *this;}

};


// namespace alps {
//   namespace alea{
//     class foo : public batch_acc<double>
//     {
//       typedef batch_acc<double> base;
//     public:
//       base b;
//       foo() : base(1), b(1){}
//       foo& operator<< (double x) {
//         static_cast<base&>(*this) << x;
//         // base::operator<<(src); return *this;
//       }
//       void add(double x){
//         // base::operator<<(x);
//         b << x;
//       }
//     };
//   }
// }

// template<typename AccType>
// typename std::enable_if<alps::alea::is_alea_acc<AccType>::value, AccType&>::type
// operator<<(AccType& acc, const typename AccType::value_type& v){
//   return acc << alps::alea::value_adapter<typename AccType::value_type>(v);
// }