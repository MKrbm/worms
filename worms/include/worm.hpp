#pragma once
#include <string.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <ostream>
#include <strstream>
#include <sstream>
#include <algorithm>
#include <utility>
#include <bcl.hpp>
#include <stdlib.h>

#ifdef __APPLE__
#  include <mach-o/dyld.h>
#endif
#if defined(_WIN32)
#  include <windows.h>
#else
  #include <unistd.h>
#endif


#include <filesystem>
#include <unistd.h>

#include <ctime>
#include <math.h> 

#include "state2.hpp"
#include "operator.hpp"
#include "automodel.hpp"
#include "funcs.hpp"
#define SEED 1662509963
/* inherit UnionFindTree and add find_and_flip function*/

// template <typename MODEL>

inline int positive_modulo(int i, int n) {
    return (i % n + n) % n;
}


using spin_state::Dotv2;

// using MODEL = model::heisenberg1D;
using STATE = spin_state::VUS;
using SPIN = spin_state::US;
using BOND = model::VS;
using WORMS = spin_state::WORM_ARR;
using DOTS = std::vector<Dotv2>;
using size_t = std::size_t;


template <class MCT>
class Worm{

  private:

  public:

  // define observables 
  double phys_cnt = 0; // number of physically meaningful configurations;
  double obs_sum = 0; // sum of observables encountered while worm update. (observable must be non-diagonal operator)
  // end of define observables


  // static const size_t sps = 2;
  // static const size_t sps_sites[site] - 1 = sps-1; // = 1 for spin half model

  typedef spin_state::Operator OP_type;
  typedef std::vector<OP_type> OPS;
  typedef spin_state::StateFunc state_func;
  using MODEL = model::base_model<MCT>;
  using LOPt = model::local_operator<MCT>;

  MODEL spin_model;
  // typedef typename base_spin_model::MCT MCT;
  OPS ops_main; //contains operators.
  OPS ops_sub; // for sub.
  STATE state;
  STATE cstate;
  DOTS spacetime_dots; //contain dots in space-time.
  WORMS worms_list;

  std::vector<BOND> bonds;
  std::vector<size_t> bond_type;
  VVS pows_vec;
  VS sps_sites;
  vector<state_func> state_funcs;
  
  std::vector<size_t> pres = std::vector<size_t>(0);
  std::vector<size_t> psop = std::vector<size_t>(0);
  // std::vector<size_t> st_cnt = std::vector<size_t>(0);
  double beta;
  size_t d_cnt=0;
  int L; //number of sites
  size_t bocnt = 0;
  size_t cutoff_length; //cut_off length
  size_t u_cnt=0;

  //declaration for random number generator
  // typedef model::local_operator::engine_type engine_type;
  typedef std::mt19937 engine_type;

  engine_type rand_src;
  engine_type test_src;
  // #ifdef NDEBUG
  // // unsigned rseed = static_cast <unsigned> (time(0));
  // // engine_type rand_src = engine_type(rseed);
  // #else
  // unsigned rseed = static_cast <unsigned> (time(0) + srand(id));
  // unsigned rseed = SEED;
  // // SEED = rseed;
  // engine_type rand_src = engine_type(SEED);
  // engine_type test_src = engine_type(SEED);
  // #endif


  // random distribution from 0 to 1
  typedef std::uniform_real_distribution<> uniform_t;
  typedef std::exponential_distribution<> expdist_t;
  uniform_t uniform;
  // reference of member variables from model class

  const int N_op;
  // std::array<model::local_operator<MODEL::MCT>, N_op>& loperators; //holds multiple local operators
  std::vector<model::local_operator<typename MODEL::MCT>>& loperators;
  std::vector<std::vector<double>> accepts; //normalized diagonal elements;
  double rho;
  int cnt=0;


  typedef bcl::markov<engine_type> markov_t;

  Worm(double beta, MODEL model_, size_t cl = SIZE_MAX, int rank = 0)
  :spin_model(model_), L(spin_model.L), beta(beta), rho(-1), N_op(spin_model.N_op), 
  bonds(spin_model.bonds),bond_type(spin_model.bond_type) ,state(spin_model.L),cstate(spin_model.L), cutoff_length(cl),
  loperators(spin_model.loperators), sps_sites(spin_model._sps_sites)
  {

    srand(rank);
    #ifdef NDEBUG
    unsigned rseed = static_cast <unsigned> (time(0)) + rand() * (rank + 1);
    rand_src = engine_type(rseed);
    #else
    rand_src = engine_type(SEED);
    test_src = engine_type(SEED);
    #endif
    double max_diagonal_weight = loperators[0].max_diagonal_weight_;
    for (auto const& lop : loperators){
      max_diagonal_weight = std::max(max_diagonal_weight, lop.max_diagonal_weight_);
    }
    for (int i=0; i<loperators.size(); i++){
      LOPt const& lop = loperators[i];
      pows_vec.push_back(lop.ogwt.pows);
      state_funcs.push_back({lop.ogwt.pows, lop.ogwt.L});
      auto accept = std::vector<double>(lop.size, 0);

      auto const& ham = lop.ham_prime;
      for (int j=0; j<lop.size; j++) {
        accept[j] = ham[j][j]/max_diagonal_weight;
      }
      accepts.push_back(accept);
      rho = max_diagonal_weight * spin_model.Nb;
  }
  }

  void initStates(){ //* initialized to all up
  for (int i=0; i<state.size(); i++){
    #ifdef RANDOM_SEED
    double r = uniform(rand_src);
    state[i] = static_cast<SPIN>(sps_sites[i] * r);
    #else
    state[i] = 0;
    #endif
    }
  }

  void initDots(bool add_state = true){
    spacetime_dots.resize(0);
    if (add_state){
      for(int i=0; i<L; i++){
        set_dots(i, -1, i);
      }
    }
  }

  //swapt main and sub
  void swapOps(){
    std::swap(ops_main, ops_sub);
  }

  //main functions

  void diagonalUpdate(double wdensity){
    dout << "random : " <<  uniform(rand_src) << endl;

    swapOps();
    // wdensity = 3;
    
    expdist_t expdist(rho * beta + wdensity); //initialize exponential distribution
    double pstart = wdensity / (beta * rho + wdensity); //probability of choosing worms
    std::copy(state.begin(), state.end(), cstate.begin());
    size_t lop_label;
    lop_label = 0; //typically, lop_label is fixed to 0
    // int leg_size = leg_sizes[lop_label]; //size of choosen operator


    ops_main.resize(0); //* init_ops_main()
    
    initDots(); //*init spacetime_dots

    //*init worms
    worms_list.resize(0);
    ops_sub.push_back(OP_type::sentinel(1)); //*sentinels
    double tau = expdist(rand_src);
    for (typename OPS::iterator opi = ops_sub.begin(); opi != ops_sub.end();){
      // auto op_sub = *opi;
      if (tau < opi->tau()){ //* if new point is behind the next operator is opsub.
        double r = uniform(rand_src);
        if (r < pstart){
          size_t s = static_cast<int>(L * uniform(rand_src));
          appendWorms(worms_list, s, spacetime_dots.size(), tau);
          set_dots(s, -2 , 0); //*index is always 0 
        }else{
          size_t b = static_cast<size_t>(bonds.size() * uniform(rand_src));
          lop_label = bond_type[b];
          auto const& accept = accepts[lop_label];
          auto const& bond = bonds[b];

          size_t u = state_funcs[lop_label].state2num(cstate, bond);
          printd("u = %lu\t", u);
          printd("input = %lu\t", u);
          printd("opi_type = %lu\n", opi->op_type());
          dout << endl;
          // size_t u = spin_state::state2num(cstate, bond);


          r = uniform(rand_src);

          if (r < accept[u]){
            dout << "append r " << r << endl;
            appendOps(ops_main, spacetime_dots, &bond, &pows_vec[lop_label] ,u * pows_vec[lop_label][bond.size()] + u, lop_label, tau);
          }
        }
        tau += expdist(rand_src);
      }else{ //*if tau went over the operator time.
        if (opi->is_off_diagonal()) {

          
          update_state(opi, cstate);
          appendOps(ops_main, spacetime_dots, opi->bond_ptr(), opi->pows_ptr(), opi->state(), opi->op_type(),opi->tau());
          printStateAtTime(cstate, tau);
        }
        ++opi;
      }
    } //end of while loop

    #ifndef NDEBUG
    for (typename OPS::iterator opi = ops_main.begin(); opi != ops_main.end();++opi){
      printf("[%d, %d]\n", opi->bond(0), opi->bond(1));
    }

    if (cstate != state){
      throw std::runtime_error("diagonalUpdate : state is not updated correctly");
    }
    #endif 
  }

  /*
  *update Worm for W times.

  variables
  ---------
  dir : direction of worm head. 1 : upward, -1 : downward
  */
  void wormUpdate(double& wcount, double& wlength){
    std::copy(state.begin(), state.end(), cstate.begin());
    pres.resize(0);
    psop.resize(0);
    for (WORMS::iterator wsi = worms_list.begin(); wsi != worms_list.end(); ++wsi){

      // t : tail, h : head. direction of tails is opposite to the direction of the initial head.
      // prime means the spin state in front of the worm.
      size_t wt_dot, site, wt_site, _t_spin, n_dot_label;
      double wt_tau, tau; // tau_prime is the time of the next operator.
      std::tie(wt_site, wt_dot , wt_tau) = *wsi; //contains site, dot label, tau
      tau = wt_tau;
      site = wt_site;
      Dotv2* dot = &spacetime_dots[wt_dot];
      Dotv2* _dot;
      double r = uniform(rand_src);
      size_t dir = (size_t)2 * r, ini_dir = dir;
      // size_t fl = 1;
      int fl = static_cast<int>((sps_sites[site]-1)*uniform(rand_src)) + 1, ini_fl = fl;
      int wl = wlength;
      int br = 0;
      bool wh = true; //* Worm head still exists.
      double wlength_prime = 0;
      wcount += 1;
      // wlength_prime = (dir == 0) ? tau : -tau;

      if (d_cnt == 54){
        int x;
        cout << d_cnt << endl;
      }

      do{
        n_dot_label = dot->move_next(dir); //next label of dot.

        size_t status = wormOpUpdate(n_dot_label, dir, site, wlength_prime, fl, tau, wt_dot, wt_site, wt_tau);
        if (status != 0){
          if (status == 1){
            wlength_prime = 0;
            reset_ops();
            std::copy(cstate.begin(), cstate.end(), state.begin());
            br = 1;
            break;
           }
          }
        dot = &spacetime_dots[n_dot_label];
      }while((n_dot_label != wt_dot || ((ini_dir == dir ? -1 : 1)*ini_fl + fl + sps_sites[site])%sps_sites[site] !=0)); 
      if(br==1){
        bocnt++;
        break;
      }

      wlength += wlength_prime;
      checkOpsInUpdate(wt_dot, dir ? n_dot_label : dot->prev(), ini_dir, ini_fl, fl, dir);
    }
  }
  /*
  This function will be called ever time the head of the worm cross the same propagation level. 
  calculate $\langle x_{\tau} | \hat{o}_i \hat{o}_j |x^\prime_{\tau} \rangle$ and update the state of the worm. $x is lower states and x^\prime is upper states$ 

  note that this function is only called when the worm position of head and tail is apart.
  params
  ------
  tau : imaginary time of the worm head and tail.
  h_site : site of the worm head.
  t_site : site of the worm tail.
  h_x : lower state of the worm head.
  h_x_prime : upper state of the worm head.
  t_w : lower state of the worm tail.
  t_x_prime : upper state of the worm tail.
  */
  

  void calcHorizontalGreen( double tau, size_t h_site, size_t t_site, 
                            int h_x, int h_x_prime, int t_x, int t_x_prime){


    // test specifically for HXXX.
    // calculate $\langle S^-_i S^+_j \rangle$
    if (h_site != t_site) {
      if (h_x == 0 && h_x_prime == 1 && t_x == 1 && t_x_prime == 0) obs_sum ++; 
      if (h_x == 1 && h_x_prime == 0 && t_x == 0 && t_x_prime == 1) obs_sum ++; 
    } else {
      if (h_x != t_x || h_x_prime != t_x_prime) throw std::runtime_error("h_x and t_x must be same.");
      if (h_x == h_x_prime) phys_cnt++;
    }
  }
  
  // //*append to ops
  static void appendOps(
    OPS& ops, 
    DOTS& sp, 
    const BOND * const bp, 
    const BOND * const pp, 
    int state, 
    int op_type, 
    double tau){

    int s = bp->size();
    ops.push_back(OP_type(bp, pp, state, op_type, tau));
    size_t n = ops.size();
    size_t label = sp.size();
    int site;
    for (int i=0; i<s; i++){
      // set_dots(bond[i], 0, i);
      site = bp->operator[](i);
      sp.push_back( Dotv2(sp[site].prev(), site, n-1, i, site));
      sp[sp[site].prev()].set_next(label);
      sp[site].set_prev(label);
      label += 1;
    }
  }
  //* get dot state
  /*
  params
  ------
  ndot_label : label of the dot worm is directing to.
  dir : direction of the worm moves to dot. 1 : upwards, 0 : downwards. So if dir = 1, it means worm comes from below the dot.
  */
  inline size_t getDotState(size_t ndot_label, size_t dir){
    Dotv2* ndot = &spacetime_dots[ndot_label];
    if (ndot->at_origin()){
      return state[ndot->label()];
    } else if (ndot->at_operator()){
      OP_type & opstate = ops_main[ndot->label()];
      size_t cindex = ndot->leg(!dir, opstate.size()); // direction must be reversed here.
      return opstate.get_local_state(cindex);
    } else if (ndot->at_worm()){
      return getDotState(ndot->move_next(dir), dir);
    } else {
      throw std::invalid_argument("dots contains invalid dot type");
      return 0;
    }
  }

  //*append to worms
  inline void appendWorms(WORMS& wm, size_t site, size_t dot_label, double tau){ 
    wm.push_back(std::make_tuple(site, dot_label, tau));
  }
 
  /*
  *perform one step from given Worm.
  If dot is operator then, Worm move to exit of the operator. otherwise just assigin spin to dot.
  params
  ------
  int next_dot : next dot.
  int dir : direction Worm is moving toward. 1 : move up, 0 : move down.
  int spin : current spin state of Worm.
  int site : site Worm is at.

  params(member variables)
  ------
  */
  int wormOpUpdate(size_t& next_dot, size_t& dir, 
                   size_t& site, double& wlength, int& fl, double& tau,
                   const size_t wt_dot, const size_t wt_site, const double wt_tau
                  ){
    
    OP_type* opsp;
    double tau_prime;
    Dotv2* dotp = &spacetime_dots[next_dot];
    
    // get imaginary temperature at the next dot.
    if (dotp->at_origin()){ tau_prime = 0; dout << "at origin" << endl;}
    else if (dotp->at_operator()){ opsp = &ops_main[dotp->label()]; tau_prime = opsp->tau();}
    else if (dotp->at_worm()){ 
      tau_prime = std::get<2>(worms_list[dotp->label()]); dout << "at worm" << endl;  
    }
    else{ throw std::runtime_error("dot is not at any of the three places"); }

    dout << tau << " " << tau_prime << " " << wt_tau << " " << dir << " passed ? " << detectWormCross(tau, tau_prime, wt_tau, dir) << endl;

    if (detectWormCross(tau, tau_prime, wt_tau, dir)){
      size_t h_x, h_x_prime, t_x, t_x_prime;
      Dotv2* wtdot = &spacetime_dots[wt_dot];

      // get spin over and under the worm head.
      t_x = getDotState(wtdot->move_next(1), 1); t_x_prime = getDotState(wtdot->move_next(0), 0);
      if (dir == 1){ 
        h_x = getDotState(next_dot, 1); h_x_prime = getDotState(dotp->move_next(0), 0);
      } else if (dir == 0){ 
        h_x = getDotState(dotp->move_next(1), 1); h_x_prime = getDotState(next_dot, 0); 
      } else { throw std::runtime_error("dir should be either 1 or 0"); }

      // if head is on tail, this case is not regarded as 2point correlation.
      calcHorizontalGreen(tau, site, wt_site, h_x, h_x_prime, t_x, t_x_prime);
    }

    size_t cur_dot = next_dot;
    // ASSERT(site == dotp->site(), "site is not consistent");
    if (dotp->at_origin()){  
      state[dotp->label()] = (state[dotp->label()] + fl) % sps_sites[site]; 
      if (dir||tau==0) wlength += 1 - tau; else wlength += tau;
    } 
    else if (dotp->at_operator()){
      dout << "update cnt : " << u_cnt << endl;
      u_cnt++;

      if (opsp->cnt()==0){
        psop.push_back(dotp->label());
        pres.push_back(opsp->state());
      }
      opsp->add_cnt();

      if (opsp->cnt() > cutoff_length){
        return 1;
      }
      
      size_t num;
      int tmp;
      int nindex;
      size_t dir_in = !dir; //n* direction the Worm comes in from the view of operator.
      size_t leg_size = opsp->size();
      size_t cindex = dotp->leg(dir_in, leg_size);
      size_t index = dotp->leg(0, leg_size);
      
      // if (u_cnt==9 || u_cnt == 2826) {
      if (fl!=0){
        opsp->update_state(cindex, fl);
        num = opsp->state();
        tmp = loperators[opsp->op_type()].markov[num](cindex*(sps_sites[site] - 1) + sps_sites[site]-fl, rand_src);
      }else{
        num = opsp->state();
        tmp = loperators[opsp->op_type()].markov[num](0, rand_src);
      }

      int tmp_wlength = opsp->tau() - tau;
      if (!(dir^(tmp_wlength>0)) & (tmp_wlength!=0)) wlength += std::abs(tmp_wlength);
      else wlength += (1 - std::abs(tmp_wlength));

      //* if Worm stop at this operator
      if (tmp == 0){        
        nindex = static_cast<size_t>((2 * leg_size)*uniform(rand_src));
        dir = nindex/(leg_size);
        site = opsp->bond(nindex%leg_size);
        fl = 0;
      }else{
        tmp--;
        nindex = tmp/(sps_sites[site] - 1);
        fl = tmp % (sps_sites[site] - 1) + 1;
        opsp->update_state(nindex, fl);
        //n* assigin for next step
        dir = nindex/(leg_size);
        site = opsp->bond(nindex%leg_size);
      }

      #ifndef NDEBUG
      int niter = 0;
      for (int i=0; i<niter; i++){
        int tmp_ = loperators[opsp->op_type()].markov[num](cindex*(sps_sites[site] - 1) + sps_sites[site]-fl-1, test_src);
        int nindex_ = tmp_/sps_sites[site] - 1;
        int fl_ = tmp_ % sps_sites[site] - 1 + 1;
        // printf("test tmp : %d, state : %d\n", tmp_, num ^ (fl_ << (nls*nindex_)));
      }
      #endif 
      next_dot = opsp->next_dot(cindex, nindex, cur_dot);
    }

    tau = tau_prime;
    return 0;
  }

  /*
  *this function will be called after assigining op_main
  */
  void set_dots(size_t site, size_t dot_type, size_t index){


    size_t label = spacetime_dots.size();

    if (dot_type == -1) {
      ASSERT(label == site, "label must be equal to site");
      spacetime_dots.push_back(
        Dotv2::state(site)
      );
    }else if(dot_type == -2){
      size_t n = worms_list.size();
      spacetime_dots.push_back(
        Dotv2::worm(spacetime_dots[site].prev(), site, n-1, site)
      );
      spacetime_dots[spacetime_dots[site].prev()].set_next(label);
      spacetime_dots[site].set_prev(label);
    }else{
      std::cout << "dot_type must be either -1 or -2" << std::endl;
    }
  }

  /*
  *this function will be called after assigining op_main
  */
  void set_op_dots(size_t site, size_t index){
    size_t label = spacetime_dots.size();
    size_t n = ops_main.size();
    spacetime_dots.push_back(
      Dotv2(spacetime_dots[site].prev(), site, n-1,index, site)
    );
    spacetime_dots[spacetime_dots[site].prev()].set_next(label);
    spacetime_dots[site].set_prev(label);
  }

  /*
  *update given state by given operator ptr;
  */
  void update_state(typename OPS::iterator opi, STATE& state){
    #ifndef NDEBUG
    STATE local_state = opi->get_state_vec();
    STATE state_(opi->size());
    int i=0;
    for (auto x : *(opi->bond_ptr())){
      state_[i] = state[x];
      i++;
    }
    ASSERT(is_same_state(local_state, state_, 0), "the operator can not be applied to the state");
    #endif
    if (opi->is_off_diagonal()) update_state_OD(opi, state);
  }

  /*
  *update given state by given offdiagonal operator ptr;
  */
  void update_state_OD(typename OPS::iterator opi, STATE& state){
    int index = 0;
    auto const& bond = *(opi->bond_ptr());
    for (auto x : bond){
      state[x] = opi->get_local_state(bond.size() + index);
      index++;
    }
  }

  /*
  * check the operator and state is consistent during the worm_updateg
  params
  ------
  worm_label : label of dot (Worm) we are aiming at.
  p_label : label of dot before reaching at the current position of Worm.

  */
  void checkOpsInUpdate(int worm_label, int p_label, int ini_dir, int ini_fl, int fl, int dir){
    
    #ifndef NDEBUG
    auto state_ = state;

    int label = 0;
    d_cnt ++;
    std::cout << "debug cnt = " << d_cnt << std::endl;
    for (const auto& dot:spacetime_dots){


      if (dot.at_operator() && (dot.leg(0, 0)==0)){ //if dot_type is operator
        auto opi = ops_main.begin() + dot.label();
        update_state(opi, state_);
      }
      else if (dot.at_origin()){
        int dot_spin = state[dot.label()];
        ASSERT(state_[dot.site()] == dot_spin, "spin is not consistent");
      }
      // else if (dot.at_worm()){
      //   // int dot_spin = std::get<1>(worms_list[dot.label()]);
      //   // int spin = (worm_label == label) ? (ini_dir^(dot_spin)) : (dot_spin);
      //   // ASSERT(state_[dot.site()] == spin, "spin is not consistent");
      //   // std::cout << "sps : " << sps << std::endl;
      //   if (worm_label == label) state_[dot.site()] = (state_[dot.site()] + ini_fl*(ini_dir ? 1 : -1)) % sps;
      // }

      // if (p_label == label){
      //   state_[dot.site()] = (state_[dot.site()] - fl * (dir ? 1 : -1))%sps;
      // }

      label++;
    }
    ASSERT(is_same_state(state_, state, 0), "operators are not consistent while update worms");
    // std::cout << "hihi" << std::endl;
    #endif 
    return;
  }

  bool detectWormCross(double tau, double tau_prime, double wt_tau, int dir){
    if (dir == 1){
      double _tau = tau_prime == 0 ? 1 : tau_prime;
      if (_tau >= wt_tau && wt_tau > tau) return true; 
      else return false;
    } else { // if dir == 0
      double _tau = tau == 0 ? 1 : tau;
      if (tau_prime <= wt_tau && wt_tau < _tau) return true;
      else return false;
    }
  }

  void reset_ops(){
    for (size_t i=0; i<psop.size(); i++){
      ops_main[psop[i]].set_state(pres[i]);
    }
  }


  bool is_same_state(int n, int m){
    return n==m;
  }

  bool is_same_state(int n, STATE state, size_t lopt){
    int m = state_funcs[lopt].state2num(state, state.size());
    return n==m;
  }

  bool is_same_state( STATE state_, STATE state, size_t lopt){
    int m = state_funcs[lopt].state2num(state, state.size());
    int n= state_funcs[lopt].state2num(state_, state.size());
    return n==m;
  }
  static void printStateAtTime(const STATE& state, double time){
    #ifndef NDEBUG
    std::cout << "current spin at time :" << time << " is : ";
    for (auto spin : state){
      std::cout << spin << " ";
    }
    std::cout << std::endl;
    #else
    return;
    #endif
  }

};

extern template class Worm<bcl::heatbath>;
extern template class Worm<bcl::st2010>;
extern template class Worm<bcl::st2013>;




