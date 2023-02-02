#include "../include/autoobservable.hpp"
#include "load_npy.hpp"


namespace model
{
    NpyWormObs::NpyWormObs(size_t sdof, size_t legs): WormObservable(sdof, legs) {
      _worm_obs.resize(pow(sdof, legs*2)); 
    }

    void NpyWormObs::_SetWormObs(std::vector<double> _obs){
      if (_worm_obs.size()!= _obs.size()){
        std::cerr << "Fail to set worm_obs due to dimension dissmatch" << std::endl;
        exit(1);
      }
      _worm_obs = std::move(_obs);
    }

    //read obs vector from numpy file.
    void NpyWormObs::ReadNpy(std::string path_to_folder){
      std::vector<std::string> path_list;
      get_npy_path(path_to_folder, path_list); //stores *.py files
      std::sort(path_list.begin(), path_list.end()); 

      return;
    }

    double NpyWormObs::_operator(size_t state) const {
      return _worm_obs[state];
    }
}

