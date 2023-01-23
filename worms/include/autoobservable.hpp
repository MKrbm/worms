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