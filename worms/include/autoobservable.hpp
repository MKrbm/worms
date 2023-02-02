#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include <array>
#include <string>
#include <numeric>
#include <random>
#include <math.h>
#include <bcl.hpp>
#include <algorithm>
#include <assert.h> 
#include <fstream>
#include <tuple>
#include <fstream>
#include "load_npy.hpp"
#include "automodel.hpp"


namespace model{
  class observable;
  class WormObservable;
  class NpyWormObs;
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



class model::WormObservable
{
  protected:
    size_t _spin_dof; // spin degree of freedom.
    size_t _leg_size; // leg size of WormObservable either 1 or 2. 1 for one-site operator and 2 for bond operators.
  public:
    WormObservable(size_t spin_dof, size_t legs): _spin_dof(spin_dof), _leg_size(legs) {
      if (legs != 1 && legs != 2) {std::cerr << "leg size of WormObservable must be 1 or 2" << std::endl; exit(1);}
    }
    size_t spin_dof() const {return _spin_dof;}
    size_t leg_size() const {return _leg_size;}
    bool is_one_site() const {return _leg_size == 1;}



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
    int get_state(std::array<size_t, 4> spins) const  {
      if (spins[0] >= _spin_dof || spins[1] >= _spin_dof || spins[2] >= _spin_dof || spins[3] >= _spin_dof) {
        std::cerr << "spin state is out of range" << std::endl; exit(1);
      }
      if (is_one_site()){
        // If heads are different, one_site_operator will return zero.
        if (spins[1] != spins[3]) return -1; 
        return spins[0] + spins[2] * _spin_dof;
      }
      size_t s = 0;
      std::array<size_t, 4>::iterator si = spins.end();
      do {
        si--;
        s *= _spin_dof;
        s += *si;
      } while (si != spins.begin());
      return s;
    }

    double operator() (std::array<size_t, 4> spins) const {
      int s = get_state(spins);
      if (s == -1) return 0;
      return _operator(s);
    }

    virtual double _operator(size_t state) const {
      std::cerr << "WormObservable::operator is virtual function" << std::endl;
      exit(1);
    }
};


class model::NpyWormObs : public WormObservable
{
  private: 
    std::vector<double> _worm_obs;
  public:
    NpyWormObs(size_t sdof, size_t legs);
    void _SetWormObs(std::vector<double> _obs);
    void ReadNpy(std::string path);
    double _operator(size_t state) const;
};



