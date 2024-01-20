#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <type_traits>
#include <string>



#include <bcl.hpp>
#include <libconfig.h++>
#include "worm.hpp"
#include "observable.hpp"
#include "automodel.hpp"
#include "autoobservable.hpp"




using namespace libconfig;
using namespace std;

// #define DEBUG 1
#define MESTIME 1

#if MESTIME
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;
#endif

extern double elapsed;



// parallel version of exe_worm
typedef std::unordered_map<std::string, model::WormObs> map_wobs_t;
template <typename MC>
std::unordered_map<std::string, model::WormObs> exe_worm_parallel(
  model::base_model<MC> spin_model, 
  double T, 
  size_t sweeps, 
  size_t therms, 
  int64_t cutoff_l, 
  bool fix_wdensity, 
  int rank,
  std::vector<batch_res>& res,
  alps::alea::autocorr_result<double>& ac_res,
  model::observable obs,
  model::MapWormObs wobs, 
  double& borate,
  int seed = -1
);


extern template map_wobs_t exe_worm_parallel<bcl::st2013>(
  model::base_model<bcl::st2013> spin_model, 
  double T, 
  size_t sweeps, 
  size_t therms, 
  int64_t cutoff_l, 
  bool fix_wdensity, 
  int rank,
  std::vector<batch_res>& res,
  alps::alea::autocorr_result<double>& ac_res,
  model::observable obs,
  model::MapWormObs wobs,
  double & borate,
  int seed = -1
);

// extern template map_wobs_t exe_worm_parallel<bcl::st2010>(
//   model::base_model<bcl::st2010> spin_model, 
//   double T, 
//   size_t sweeps, 
//   size_t therms, 
//   size_t cutoff_l, 
//   bool fix_wdensity, 
//   int rank,
//   std::vector<batch_res>& res,
//   alps::alea::autocorr_result<double>& ac_res,
//   model::observable obs,
//   model::MapWormObs wobs,
//   double& borate,
//   int seed = -1
// );

extern template map_wobs_t exe_worm_parallel<bcl::heatbath>(
  model::base_model<bcl::heatbath> spin_model, 
  double T, 
  size_t sweeps, 
  size_t therms, 
  int64_t cutoff_l, 
  bool fix_wdensity, 
  int rank,
  std::vector<batch_res>& res,
  alps::alea::autocorr_result<double>& ac_res,
  model::observable obs,
  model::MapWormObs wobs,
  double& borate,
  int seed = -1
);


