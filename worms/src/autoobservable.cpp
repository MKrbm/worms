#include <fstream>

#include "../include/autoobservable.hpp"
#include "load_npy.hpp"

using namespace std;

bool endsWith(const string &str, const string &suffix) {
    return str.size() >= suffix.size() &&
           str.rfind(suffix) == str.size() - suffix.size();
}

bool fileExists(const string &path) {
    ifstream file(path);
    return file.good();
}

namespace model
{
    NpyWormObs::NpyWormObs(size_t sdof, size_t legs): WormObservable(sdof, legs){}
    void NpyWormObs::_SetWormObs(vector<double> _obs){
      if (_worm_obs.size() != 0){
        cerr << "Worm_obs is already set" << endl;
        exit(1);
      }
      if (pow(_spin_dof, _leg_size*2)!= _obs.size()){
        cerr << "Fail to set worm_obs due to dimension dissmatch" << endl;
        exit(1);
      }
      _worm_obs = move(_obs);
    }

    //read obs vector from numpy file.
    /*
                x_t'         x_h'
    weight of   2 points ops (O)     =   <x' | O | x >   = O_{x', x}
                x_t          x_h

    x = x_t + x_h * S
    {x', x} -> x' * S^2  + x

    */
    void NpyWormObs::ReadNpy(string file_path){
      if (!endsWith(file_path, ".npy")){
        cerr << "File path is not *.npy" << endl;
        exit(1);
      }

      if (!fileExists(file_path)){
        cerr << "File does not exist" << endl;
        exit(1);
      }

      pair<vector<size_t>, vector<double>> pair;
      try {pair = load_npy(file_path);}
      catch (const std::exception& e) {
        cerr << "Fail to load npy file with unknown reason" << endl;
        exit(1);
      }

      vector<size_t> shape = pair.first;
      vector<double> data = pair.second;

      if (shape.size() != 2){
        cerr << "Require 2D array" << endl;
        exit(1);
      }

      _SetWormObs(data);

      return;
    }

    double NpyWormObs::_operator(size_t state) const {
      return _worm_obs[state];
    }
}

