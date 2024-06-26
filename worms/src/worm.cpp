#include "../include/worm.hpp"

double getWormDistTravel(double tau, double tau_prime, int dir) {
  if (dir == 1) {
    tau_prime = tau_prime == 0 ? 1 : tau_prime;
    return tau_prime - tau;
  } else {
    tau == 0 ? tau = 1 : tau;
    return tau - tau_prime;
  }
}

template <class MCT>
Worm<MCT>::Worm(double beta, MODEL model_, model::MapWormObs mp_worm_obs_,
                size_t cl, int rank, int seed)
    : spin_model(model_),
      L(spin_model.L),
      beta(beta),
      rho(-1),
      N_op(spin_model.N_op),
      bonds(spin_model.bonds),
      bond_type(spin_model.bond_type),
      state(spin_model.L),
      cstate(spin_model.L),
      cutoff_length(cl),
      _mp_worm_obs(mp_worm_obs_),
      _phys_cnt(1),
      cutoff_thres(1000),
      loperators(spin_model.loperators),
      sps(spin_model.sps_sites(0)),
      obs_sum(mp_worm_obs_().size()),
      zw(spin_model.zw) {
  srand(rank);
#ifdef NDEBUG
  unsigned rseed = static_cast<unsigned>(time(0)) + rand() * (rank + 1);
  unsigned engine_seed;
  if (seed < 0) {
    engine_seed = rseed;
  } else {
    engine_seed = seed + (rank + 1);
  }
  std::cout << "Random seed for process : "<< rank << " is set to : " << engine_seed << std::endl;
  rand_src = engine_type(engine_seed);

#else
  if (seed < 0) {
    seed = SEED;
  }
  std::cout << "Simulation random seed was set to : " << seed << std::endl;
  rand_src = engine_type(seed);
#endif

  max_diagonal_weight = 0;
  for (auto const &lop : loperators) {
    max_diagonal_weight =
        std::max(max_diagonal_weight, lop.max_diagonal_weight_);
  }

  for (double s_flip_max : spin_model.s_flip_max_weights) {
    max_diagonal_weight = std::max(max_diagonal_weight, s_flip_max);
  }

  for (int i = 0; i < loperators.size(); i++) {
    LOPt const &lop = loperators[i];
    pows_vec.push_back(lop.ogwt.pows);
    state_funcs.push_back({sps, lop.ogwt.L});
    auto accept = std::vector<double>(lop.size, 0);

    auto &ham = lop.ham_vector();
    for (int j = 0; j < lop.size; j++) {
      accept[j] = ham[j * lop.size + j] / max_diagonal_weight;
    }
    accepts.push_back(accept);
  }

  for (int i = 0; i < num_type(spin_model.site_type); i++) {
    pows_vec.push_back(vector<size_t>{1, sps, sps * sps});
  }

  rho = max_diagonal_weight * (spin_model.Nb + spin_model.L);

  // d* define nn_sites in the same manner as bonds
  nn_sites.resize(spin_model.L);
  for (size_t site = 0; site < spin_model.L; site++) {
    for (model::BondTargetType bt : spin_model.nn_sites[site]) {
      nn_sites[site].push_back(bt.target);
    }
  }
}
template <class MCT>
void Worm<MCT>::diagonalUpdate(double wdensity) {
  dout << "hello" << endl;
  swapOps();
  expdist_t expdist(rho * beta +
                    wdensity);  // initialize exponential distribution
  double pstart =
      wdensity / (beta * rho + wdensity);  // probability of choosing worms
  std::copy(state.begin(), state.end(), cstate.begin());
  size_t lop_label;
  lop_label = 0;  // typically, lop_label is fixed to 0

  ops_main.resize(0);    //* init_ops_main()
  can_warp_ops.clear();  // forget previous record

  initDots();  //*init spacetime_dots

  // n*init worms
  worms_list.resize(0);
  worm_states.resize(0);
  worm_taus.resize(0);

  ops_sub.push_back(OP_type::sentinel(1));  //*sentinels
  dout << ops_sub.size() << endl;
  double tau0 = uniform(rand_src);
  double tau = expdist(rand_src);
  bool set_atleast_one = false;

  dout << tau0 << " " << tau << std::endl;
  dout << "debug count : " << cnt << std::endl;
  cnt++;
  // if (cnt == 6) {
  //   exit(1);
  // }

  state_t nn_state;
  for (typename OPS::iterator opi = ops_sub.begin(); opi != ops_sub.end();) {
    if (tau0 < tau && !set_atleast_one) {
      if (tau0 < opi->tau()) {
        size_t s = static_cast<int>(L * uniform(rand_src));
        appendWorms(worms_list, s, spacetime_dots.size(), tau0);
        set_dots(s, -2, 0);  //*index is always 0
        set_atleast_one = true;
        worm_states.push_back(cstate);
        worm_taus.push_back(tau0);
      }
    }
    if (tau <
        opi->tau()) {  //* if new point is behind the next operator is opsub.
      double r = uniform(rand_src);
      if (r < pstart) {
        size_t s = static_cast<int>(L * uniform(rand_src));
        appendWorms(worms_list, s, spacetime_dots.size(), tau);
        set_dots(s, -2, 0);  //*index is always 0
        worm_states.push_back(cstate);
        worm_taus.push_back(tau);
      } else {
        size_t b = static_cast<size_t>((spin_model.Nb + spin_model.L) *
                                       uniform(rand_src));
        r = uniform(rand_src);
        if (b < spin_model.Nb) {
          double bop_label = bond_type[b];
          auto const &accept = accepts[bop_label];
          auto const &bond = bonds[b];
          size_t u = state_funcs[bop_label].state2num(cstate, bond);
          if (accept[u] > 1) throw std::runtime_error("accept[u] > 1");
          if (r < accept[u]) {
            appendOps(ops_main, spacetime_dots, can_warp_ops, &bond,
                      &pows_vec[bop_label],
                      u * pows_vec[bop_label][bond.size()] + u, bop_label, tau);
          }
        } else {
          int site = b - spin_model.Nb;
          double sop_label = spin_model.site_type[site];
          size_t spin = cstate[site];
          double mat_elem =
              get_single_flip_elem(site, spin, spin, cstate, nn_state);
          mat_elem = std::abs(mat_elem) / max_diagonal_weight;
          if (mat_elem > 1) throw std::runtime_error("mat_elem > 1");
          if (r < mat_elem) {
            appendSingleOps(ops_main, spacetime_dots, can_warp_ops, site,
                            &nn_sites[site],
                            &pows_vec[sop_label + loperators.size()],
                            spin + spin * sps, nn_state, -1, tau);
          }
        }
      }
      tau += expdist(rand_src);
    } else {  //*if tau went over the operator time.
      if (opi->is_off_diagonal()) {
        update_state(opi, cstate);
        if (opi->_check_is_bond()) {
          appendOps(ops_main, spacetime_dots, can_warp_ops, opi->bond_ptr(),
                    opi->pows_ptr(), opi->state(), opi->op_type(), opi->tau());
        } else {
          // d* opi is a single site operator
          ptrdiff_t _index = opi->bond_ptr() - &nn_sites[0];
          size_t index = static_cast<size_t>(_index);
#ifndef NDEBUG
          if ((*opi->bond_ptr()) != nn_sites[index]) {
            throw std::runtime_error("index is wrong");
          }
#endif
          appendSingleOps(ops_main, spacetime_dots, can_warp_ops, index,
                          opi->bond_ptr(), opi->pows_ptr(), opi->state(),
                          opi->nn_state(), opi->op_type(), opi->tau());
        }
      }
      ++opi;
    }
  }  // end of while loop
#ifndef NDEBUG
  if (cstate != state) {
    throw std::runtime_error("diagonalupdate : state is not updated correctly");
  }
#endif
}

/*
*update Worm for W times.
variables
---------
dir : direction of worm head. 1 : upward, -1 : downward
*/
template <class MCT>
void Worm<MCT>::wormUpdate(double &wcount, double &wlength, size_t &w_update_cnt, const size_t cutoff_thres) {
  pops_main.resize(0);
  psop.resize(0);
  ops_copy.clear(); 
  std::copy(ops_main.begin(), ops_main.end(), std::back_inserter(ops_copy));
  std::copy(state.begin(), state.end(), cstate.begin());
  auto copystate = state;
  auto copysign = sign;
  ops_main.push_back(OP_type::sentinel(1));  //*sentinels
  typename WORMS::iterator wsi = worms_list.begin();
  size_t w_index = 0;
  for (typename OPS::iterator opi = ops_main.begin(); opi != ops_main.end();) {
    if (opi->tau() <
        std::get<2>(*wsi)) {  // n* if operator is behind the worm tail.
#ifndef NDEBUG
      if (opi->is_off_diagonal()) {
        update_state(opi, cstate);
      }
#endif
      ++opi;
    } else {
#ifndef NDEBUG
      if (cstate != worm_states[w_index]) {
        throw std::runtime_error("wormUpdate : state is not updated correctly");
      }
#endif
      // t : tail, h : head. direction of tails is opposite to the direction of
      // the initial head. prime means the spin state in front of the worm.
      int wt_dot;
      int site;
      int wt_site;
      int _t_spin;
      int n_dot_label;
      int fl = static_cast<int>((sps - 1) * uniform(rand_src)) + 1;
      int t_fl = fl;
      double wt_tau;
      double tau;  // tau_prime is the time of the next operator.
      double r = uniform(rand_src);
      int dir = static_cast<size_t>(2) * r;
      int t_dir = 1 - dir;
      std::tie(wt_site, wt_dot, wt_tau) =
          *wsi;  // contains site, dot label, tau
      tau = wt_tau;
      site = wt_site;
      Dotv2 *dot = &spacetime_dots[wt_dot];
      Dotv2 *_dot;
      bool br = false;

      dout << "r : " << r << std::endl;
      // n* spin state at worm head
      worm_states[w_index][wt_site] =
          (worm_states[w_index][wt_site] + fl) % sps;
      int w_x = worm_states[w_index][wt_site];
      int wt_x = w_x;
      bool wh = true;  //* Worm head stil exists.
      double wlength_prime = 0;
      wcount += 1;

      // n* initialize wobs variables
      phys_cnt = 0;
      obs_sum.assign(obs_sum.size(), 0);

      do {
        if (fl != 0) {
          n_dot_label = dot->move_next(dir);  // next label of dot.
        }
        if (w_update_cnt > cutoff_thres) {
          ops_main.clear();
          std::copy(ops_copy.begin(), ops_copy.end(), std::back_inserter(ops_main));
          std::copy(copystate.begin(), copystate.end(), state.begin());
          sign = copysign;
          /*  
          //n: 
            We don't need to reset 
            1. cstate
            2. worm_states
            because they will be reset in diagonalUpdate.
          */
          bocnt++;
          // std::cout << "break" << std::endl;
          wlength += wlength_prime; 
          return;
        }
        size_t status =
            wormOpUpdate(n_dot_label, dir, site, wlength_prime, fl, tau, wt_dot,
                         wt_site, wt_tau, w_x, wt_x, t_fl, t_dir, w_index);

        u_cnt++;
        if (fl != 0) {
          dot = &spacetime_dots[n_dot_label];
        }
        w_update_cnt++;
      } while ((n_dot_label != wt_dot ||
                ((t_dir == dir ? 1 : -1) * t_fl + fl + sps) % sps != 0));

      // n* undo unnecessary flip.
      if (dir != t_dir) {
        worm_states[w_index][site] =
            (worm_states[w_index][site] + sps - fl) % sps;
      }

      // n* debug
#ifndef NDEBUG
      if (cstate != worm_states[w_index]) {
        throw std::runtime_error("wormUpdate : state is not updated correctly");
      }
#endif

      // n* caluclation contribution to wobs
      _phys_cnt << phys_cnt;
      int obs_i = 0;
      for (auto &obs : _mp_worm_obs()) {
        obs.second << obs_sum[obs_i];
        obs_i++;
      }

      wlength += wlength_prime;
      checkOpsInUpdate(wt_dot, dir ? n_dot_label : dot->prev(), t_dir, t_fl, fl,
                       dir);
      ++wsi;
      ++w_index;
    }
    if (wsi == worms_list.end()) {
      break;
    }
  }
#ifndef NDEBUG
  if (cstate != worm_states[w_index - 1]) {
    throw std::runtime_error(
        "wormUpdate : state and cstate must be same after the worm update");
  }
#endif
  ops_main.resize(ops_main.size() - 1);
}

/*
This function will be called ever time the head of the worm cross the same
propagation level. calculate $\langle x_{\tau} | \hat{o}_i \hat{o}_j
|x^\prime_{\tau} \rangle$ and update the state of the worm. $x is lower states
and x^\prime is upper states$

note that this function is only called when the worm position of head and tail
is apart. params
------
tau : imaginary time of the worm head and tail.
h_site : site of the worm head.
t_site : site of the worm tail.
h_x : lower state of the worm head.
h_x_prime : upper state of the worm head.
t_w : lower state of the worm tail.
t_x_prime : upper state of the worm tail.
*/
template <class MCT>
void Worm<MCT>::calcHorizontalGreen(double tau, size_t h_site, size_t t_site,
                                    size_t h_x, size_t h_x_prime, size_t t_x,
                                    size_t t_x_prime, const state_t &_cstate) {
  int j = 0;
  for (auto &obs : _mp_worm_obs()) {
    auto &_worm_obs = obs.second;
    if (h_site != t_site) {
      obs_sum[j] +=
          _worm_obs.second()->operator()(t_x, h_x, t_x_prime, h_x_prime) * L *
          sign / 2.0;
    } else {
      if (t_x == t_x_prime) {
        for (size_t i = 0; i < L; i++) {
          size_t h_x = _cstate[i];
          if (i == t_site) {
            obs_sum[j] += _worm_obs.first()->operator()(t_x, t_x) * L * sign /
                          (double)(sps - 1);
          } else {
            obs_sum[j] += _worm_obs.second()->operator()(t_x, h_x, t_x, h_x) *
                          L / 2.0 * sign / (double)(sps - 1);
          }
        }
        phys_cnt = (double)sign / (sps - 1);
      } else {  // n* This case could contribute to single flip operator but not
                // implemented yet.
        ;
      }
    }
    j++;
  }
}

/*
This function will be called ever time worm head warps.
*/
template <class MCT>
void Worm<MCT>::calcWarpGreen(double tau, size_t t_site, size_t t_x,
                              size_t t_x_prime, const state_t &_cstate) {
  int j = 0;
  for (auto &obs : _mp_worm_obs()) {
    auto &_worm_obs = obs.second;
    if (t_x == t_x_prime) {
      throw std::runtime_error(
          "t_x == t_x_prime while wapr should never happen");
    }
    // int i = uniform(rand_src) * L;
    for (size_t i = 0; i < L; i++) {
      size_t h_x = _cstate[i];
      if (i == t_site) {
        obs_sum[j] +=
            _worm_obs.first()->operator()(t_x, t_x_prime) * L * sign * 2;
      } else {
        obs_sum[j] += _worm_obs.second()->operator()(t_x, h_x, t_x_prime, h_x) *
                      L * sign * 2;
      }
    }
    j++;
  }
}

// //*append to ops
template <class MCT>
void Worm<MCT>::appendOps(OPS &ops, DOTS &sp,
                          std::unordered_set<size_t> &warp_sp,
                          const BOND *const bp, const BOND *const pp, int state,
                          int op_type, double tau) {
  int s = bp->size();
  if (op_type >= int(loperators.size()) || op_type < 0) {
    // n* output the value op_type >= loperators.size()

    std::string msg = "op_type is out of range op_type >= loperators.size() :" +
                      std::to_string(op_type >= loperators.size());

    throw std::runtime_error(msg);
  }
  if (2 * bp->size() != pp->size() - 1)
    throw std::runtime_error("bond_ptr and pows_ptr doesn't match");
  ops.push_back(OP_type(bp, pp, state, op_type, tau));

  size_t n = ops.size();
  size_t label = sp.size();
  int site;

  //! warp is disabled for now
  if (loperators[op_type].has_warp(state)) {
    warp_sp.insert(label);
  }  // if the operator has warp, add the label of the leftmost dot to the set.

  for (int i = 0; i < s; i++) {
    // set_dots(bond[i], 0, i);
    site = bp->operator[](i);
    sp.push_back(Dotv2(sp[site].prev(), site, n - 1, i, site));
    sp[sp[site].prev()].set_next(label);
    sp[site].set_prev(label);
    label += 1;
  }
}

// //*append single-operator to ops
template <class MCT>
void Worm<MCT>::appendSingleOps(OPS &ops, DOTS &sp,
                                std::unordered_set<size_t> &warp_sp, int s_site,
                                const BOND *const bp, const BOND *const pp,
                                int state, const state_t &nn_state, int op_type,
                                double tau) {
  if (op_type >= 0) {
    // n* output the value op_type >= loperators.size()

    std::string msg = "op_type must be negative for single-flip operator";
    throw std::runtime_error(msg);
  }

#ifndef NDEBUG
  if (nn_state.size() != bp->size())
    throw std::runtime_error("nn_state.size() != bp->size()");
  for (size_t i = 0; i < bp->size(); i++) {
    if (nn_state.at(i) != cstate[bp->at(i)]) {
      throw std::runtime_error("nn_state[site] != state");
    }
  }
#endif

  ops.push_back(OP_type(bp, pp, state, nn_state, op_type, tau));

  size_t n = ops.size();
  size_t label = sp.size();
  int site;
  warp_sp.insert(label);
  for (int i = 0; i < bp->size() + 1; i++)  // nn_site + center site
  {
    if (i == 0)
      site = s_site;
    else
      site = bp->at(i - 1);
    sp.push_back(Dotv2(sp[site].prev(), site, n - 1, i, site));
    sp[sp[site].prev()].set_next(label);
    sp[site].set_prev(label);
    label += 1;
  }

  //! warp is disabled for now
  // if (loperators[op_type].has_warp(state))
  // {
  //   warp_sp.insert(sp.size());
  // } // if the operator has warp, add the label of the leftmost dot to the
  // set.
}
//* get dot state
/*
params
------
ndot_label : label of the dot worm is directing to.
dir : direction of the worm moves to dot. 1 : upwards, 0 : downwards. So if dir
= 1, it means worm comes from below the dot.
*/
template <class MCT>
size_t Worm<MCT>::getDotState(size_t ndot_label, size_t dir) {
  Dotv2 *ndot = &spacetime_dots[ndot_label];
  if (ndot->at_origin()) {
    return state[ndot->label()];
  } else if (ndot->at_operator()) {
    OP_type &opstate = ops_main[ndot->label()];
    if (opstate._check_is_bond()) {
      size_t cindex =
          ndot->leg(!dir, opstate.size());  // direction must be reversed here.
      return opstate.get_local_state(cindex);
    } else {
      //* if the operator is single-flip operator
      int index = ndot->index();
      if (index == 0)
        return opstate.get_local_state(!dir);
      else {
        return opstate.nn_state(index - 1);
      }
    }
  } else if (ndot->at_worm()) {
    return getDotState(ndot->move_next(dir), dir);
  } else {
    throw std::invalid_argument("dots contains invalid dot type");
    return 0;
  }
}

/*
*perform one step from given Worm.
If dot is operator then, Worm move to exit of the operator. otherwise just
assigin spin to dot. params
------
int next_dot : next dot.
int dir : direction Worm is moving toward. 1 : move up, 0 : move down.
int spin : current spin state of Worm.
int site : site Worm is at.

params(member variables)
------
*/
template <class MCT>
int Worm<MCT>::wormOpUpdate(int &next_dot, int &dir, int &site, double &wlength,
                            int &fl, double &tau, const int wt_dot,
                            const int wt_site, const double wt_tau, int &w_x,
                            int &wt_x, const int t_fl, const int t_dir,
                            const int w_index) {
  dout << "next_dot, dir, site, fl, tau, wt_dot, wt_site, wt_tau, w_x, "
          "wt_x, t_fl, t_dir, w_index : "
       << next_dot << " " << dir << " " << site << " " << fl << " " << tau
       << " " << wt_dot << " " << wt_site << " " << wt_tau << " " << w_x << " "
       << wt_x << " " << t_fl << " " << t_dir << " " << w_index << std::endl;
  if (fl != 0 && (w_x == -1 || site == -1 || dir < 0)) {
    throw std::runtime_error(
        "warm is in warp state but some variables are not ready for that");
  }
  OP_type *opsp;
  double tau_prime;
  size_t h_x, h_x_prime, t_x, t_x_prime;
  size_t cur_dot;
  int op_type;
  bool warped = false;
  Dotv2 *dotp = &spacetime_dots[next_dot];
  Dotv2 *wtdot = &spacetime_dots[wt_dot];
  std::unordered_set<size_t>::iterator it;
  dout << "label" << dotp->label() << std::endl;
  if (u_cnt == 105) {
    int x;
  }
  // get imaginary temperature at the next dot.
  if (fl == 0) {
    if (w_x != -1 || site != -1 || dir >= 0) {
      throw std::runtime_error(
          "warm is in warp state but some variables are not ready for that");
    }
    size_t n = static_cast<size_t>(can_warp_ops.size() * uniform(rand_src));
    it = std::begin(can_warp_ops);
    std::advance(it, n);
    cur_dot = next_dot = *it;
    dotp = &spacetime_dots[next_dot];
    opsp = &ops_main[dotp->label()];
    op_type = opsp->op_type();
    dout << "n, cur_dot, next_dot, op_type : " << n << " " << cur_dot << " "
         << next_dot << " " << op_type << std::endl;
    if (opsp->op_type() >= 0) {
      goto warp_label;  // check if optype is single flip operator
    } else {
      goto single_warp;  // check if optype is bond operator
    }
  } else if (dotp->at_origin()) {
    tau_prime = 0;
  } else if (dotp->at_operator()) {
    opsp = &ops_main[dotp->label()];
    tau_prime = opsp->tau();
  } else if (dotp->at_worm()) {
    tau_prime = std::get<2>(worms_list[dotp->label()]);
  } else {
    throw std::runtime_error("dot is not at any of the three places");
  }

  double wlength_tmp;
  wlength_tmp = getWormDistTravel(tau, tau_prime, dir);
  dout << "wlength_tmp, tau, tau_prime, dir : " << wlength_tmp << " " << tau
       << " " << tau_prime << " " << dir << std::endl;
  if (wlength_tmp < 0) {
    throw std::runtime_error("worm length is negative");
  }
  wlength += wlength_tmp;

  // getSpinsDot(next_dot, dotp, dir, h_x, h_x_prime);
  dout << worm_taus << std::endl;
  for (size_t i = 0; i < worm_taus.size(); i++) {
    if (detectWormCross(tau, tau_prime, worm_taus[i], dir)) {
      if (i == w_index) {
        if (abs(worm_taus[i] - wt_tau) > 1E-10) {
          throw std::runtime_error("worms are crossing each other");
        }
        if (wt_site == site) {
          int _fl = (dir == t_dir ? fl + t_fl : sps + fl - t_fl) % sps;
          h_x = dir == 1 ? w_x : (sps + w_x - _fl) % sps;
          h_x_prime = dir == 0 ? w_x : (sps + w_x - _fl) % sps;
          t_x = h_x;
          t_x_prime = h_x_prime;
        } else {
          h_x = dir == 1 ? w_x : (sps + w_x - fl) % sps;
          h_x_prime = dir == 0 ? w_x : (sps + w_x - fl) % sps;
          t_x = t_dir == 1 ? wt_x : (sps + wt_x - t_fl) % sps;
          t_x_prime = t_dir == 0 ? wt_x : (sps + wt_x - t_fl) % sps;
        }

#ifndef NDEBUG
        size_t _h_x, _h_x_prime, _t_x, _t_x_prime;
        _t_x = getDotState(wtdot->move_next(0), 0);
        _t_x_prime = getDotState(wtdot->move_next(1), 1);
        if (dir == 1) {
          _h_x_prime = getDotState(next_dot, 1);
          _h_x = getDotState(dotp->move_next(0), 0);
        } else if (dir == 0) {
          _h_x_prime = getDotState(dotp->move_next(1), 1);
          _h_x = getDotState(next_dot, 0);
        } else {
          throw std::runtime_error("dir should be either 1 or 0");
        }

        if (h_x != _h_x || h_x_prime != _h_x_prime || t_x != _t_x ||
            t_x_prime != _t_x_prime) {
          throw std::runtime_error("spin is not consistent");
        }
        dout << "h_x : " << h_x << " h_x_prime : " << h_x_prime
             << " t_x : " << t_x << " t_x_prime : " << t_x_prime << std::endl;
#endif
        calcHorizontalGreen(tau, site, wt_site, h_x, h_x_prime, t_x, t_x_prime,
                            worm_states[w_index]);
      }
      if (i == w_index) {
        if (wt_site != site && (worm_states[i][site] + fl) % sps != w_x) {
          throw std::runtime_error("spin is not consistent");
        }
        if (wt_site == site &&
            (worm_states[i][site] + (t_dir != dir ? sps + fl - t_fl : fl)) %
                    sps !=
                w_x) {
          throw std::runtime_error("spin is not consistent");
        }
      }
      // worm_states[i][site] = ( i == w_index) ? w_x : (worm_states[i][site] +
      // fl) % sps; // n* assign new spin.
      worm_states[i][site] = (worm_states[i][site] + fl) % sps;

      // if (i != w_index && worm_states[i][site] != w_x)  {
      //   throw std::runtime_error("spin is not consistent");
      // }
      if (fl == 0) {
        throw std::runtime_error(
            "fl must be non-zero since zero worm doesn't come here");
      }
      if (i == w_index) {
        cstate[site] = w_x;
        if (wt_site == site) {
          wt_x = (wt_x + fl) % sps;  // n* fliped by worm head.
          w_x = (t_dir == dir) ? (w_x + sps - t_fl) % sps : (w_x + t_fl) % sps;
          if (w_x != ((dir == 1 ? h_x_prime : h_x) + fl) % sps) {
            throw std::runtime_error("w_x is not consistent");
          }
        } else {
        }
      }
    }
  }

  cur_dot = next_dot;
  if (dotp->at_origin()) {
    state[dotp->label()] = (state[dotp->label()] + fl) % sps;
  } else if (dotp->at_operator()) {
    int state_num;
    int tmp;
    int nindex;
    int dir_in;
    int leg_size;
    int cindex;
    int index;
    dir_in = 1-dir;  // n* direction the Worm comes in from the view of operator.
    leg_size = opsp->size();
    cindex = dotp->leg(dir_in, leg_size);
    index = dotp->leg(0, leg_size);
    op_type = opsp->op_type();
    dout << "dir, leg_size, cindex, index, op_type : " << dir << " " << leg_size
         << " " << cindex << " " << index << " " << op_type << std::endl;

    dout << "op_type : " << op_type << std::endl;
    if (op_type < 0) {  // if the operator is single-flip operator
    single_warp:
      int nn_index = dotp->index();
      if (nn_index == 0) {
        int num = opsp->state();
        auto flip = markov_next_flip(*opsp, fl ? dir_in : 0, fl, zw);
        dir = flip.first;
        fl = flip.second;
        w_x = opsp->get_local_state(dir);
        site = static_cast<int>(opsp->bond_ptr() - &nn_sites[0]);
        if (fl == 0) {
          w_x = -1;  // w_x is ready to warp.
          site = -1;
          dir = -2;
        }
        dout << "nn_index, num, dir, fl, w_x, site : " << nn_index << " " << num
             << " " << dir << " " << fl << " " << w_x << " " << site
             << std::endl;
      } else {
        if (fl == 0) {
          std::runtime_error(
              "fl must be non-zero since zero worm doesn't come "
              "out from operator");
        }
        auto flip = markov_diagonal_nn(*opsp, dir_in, fl, nn_index - 1);
        dir = flip.first;
        fl = flip.second;
        w_x = opsp->nn_state(nn_index - 1);

        dout << "nn_index, dir, fl, w_x : " << nn_index << " " << dir << " "
             << fl << " " << w_x << std::endl;
      }
      tau = opsp->tau();
      return 0;
    }

    sign *= loperators[op_type].signs[opsp->state()];  //! sign for single-flip
                                                       //! is not included yet.
    if (true) {
      state_num = opsp->update_state(cindex, fl);
      dout << "cindex : " << cindex << " state_num : " << state_num
           << " op_type : " << op_type << std::endl;
      dout << " fl : " << fl << " sps : " << sps << std::endl;
      tmp = loperators[op_type].markov[state_num](cindex * (sps - 1) + sps - fl,
                                                  rand_src);
    } else {
    warp_label:
      if (fl != 0)
        throw std::runtime_error("fl must be zero when warp_label is called");
      warped = true;
      state_num = opsp->state();
      tmp = loperators[op_type].markov[state_num](0, rand_src);
      dout << "state_num, tmp : " << state_num << " " << tmp << std::endl;
    }
    // n* if Worm stop at this operator
    if (tmp == 0) {  // if head will warp.
      if (!loperators[op_type].has_warp(state_num)) {
        can_warp_ops.erase(cur_dot - dotp->index());
      } else {
        can_warp_ops.insert(cur_dot - dotp->index());
      }
      if (!warped) {
        sign *= loperators[op_type]
                    .signs[state_num];  // include an effect by start point
      }
      t_x_prime = getDotState(wtdot->move_next(1), 1);
      t_x = getDotState(wtdot->move_next(0), 0);
      if (it == can_warp_ops.begin()) {
        calcWarpGreen(tau, wt_site, t_x, t_x_prime, worm_states[w_index]);
      }
      fl = 0;    // redo selection of warping point.
      w_x = -1;  // w_x is ready to warp.
      site = -1;
      dir = -1;
      tau = opsp->tau();
      dout << "tmp : " << tmp << " nindex : " << nindex
           << " state_num : " << state_num << " fl : " << fl
           << " sign : " << sign << std::endl;
      return 0;
    } else {
      leg_size = opsp->size();
      tmp--;
      nindex = tmp / (sps - 1);
      if (warped) {
        if (fl != 0) {
          throw std::runtime_error("fl must be zero when warp_label is called");
        }
        sign *= loperators[op_type].signs[opsp->state()];
      }
      fl = tmp % (sps - 1) + 1;
      state_num = opsp->update_state(nindex, fl);
      sign *= loperators[op_type].signs[state_num];
      dout << "tmp was not 0 originally" << std::endl;
      dout << "tmp : " << tmp << " nindex : " << nindex
           << " state_num : " << state_num << " fl : " << fl
           << " sign : " << sign << std::endl;
      // n* assigin for next step
      dir = nindex / (leg_size);
      site = opsp->bond(nindex % leg_size);
      dout << "nindex, leg_size, dir, site : " << nindex << " " << leg_size
           << " " << dir << " " << site << std::endl;
      if (warped) {
        next_dot = opsp->next_dot(0, nindex, cur_dot);
        tau_prime = opsp->tau();
      } else {
        next_dot = opsp->next_dot(cindex, nindex, cur_dot);
      }
      if (!loperators[op_type].has_warp(state_num)) {
        can_warp_ops.erase(cur_dot - dotp->index());
      } else {
        can_warp_ops.insert(cur_dot - dotp->index());
      }

      dout << "tau_prime : " << tau_prime << std::endl;
    }

    w_x = opsp->get_local_state(nindex);

#ifndef NDEBUG

    int dsign = 1;
    for (auto &op : ops_main) {
      if (op.op_type() == -1) {
        dsign *= get_single_flip_elem(op) >= 0 ? 1 : -1;
      } else {
        dsign *= loperators[op.op_type()].signs[op.state()];
      }
    }
    if (dsign != sign) {
      std::cerr << "sign is wrong" << endl;
      exit(1);
    }
#endif
  }

  tau = tau_prime;
  return 0;
}

/*
 *this function will be called after assigining op_main
 */
template <class MCT>
void Worm<MCT>::set_dots(size_t site, size_t dot_type, size_t index) {
  size_t label = spacetime_dots.size();

  if (dot_type == -1) {
    ASSERT(label == site, "label must be equal to site");
    spacetime_dots.push_back(Dotv2::state(site));
  } else if (dot_type == -2) {
    size_t n = worms_list.size();
    spacetime_dots.push_back(
        Dotv2::worm(spacetime_dots[site].prev(), site, n - 1, site));
    spacetime_dots[spacetime_dots[site].prev()].set_next(label);
    spacetime_dots[site].set_prev(label);
  } else {
    std::cout << "dot_type must be either -1 or -2" << std::endl;
  }
}

template <class MCT>
void Worm<MCT>::getSpinsDot(size_t next_dot, Dotv2 *dotp, int dir, size_t &h_x,
                            size_t &h_x_prime) {
  if (dir == 1) {
    h_x_prime = getDotState(next_dot, 1);
    h_x = getDotState(dotp->move_next(0), 0);
  } else if (dir == 0) {
    h_x_prime = getDotState(dotp->move_next(1), 1);
    h_x = getDotState(next_dot, 0);
  }
}

/*
 *update given state by given operator ptr;
 */
template <class MCT>
void Worm<MCT>::update_state(typename OPS::iterator opi, state_t &state) {
#ifndef NDEBUG

  //* define local_state as nn_state for single site operator, bond_ptr is also
  // list of nn_sites.
  state_t local_state =
      opi->_check_is_bond() ? opi->get_state_vec() : opi->nn_state();
  state_t state_(opi->size());
  int i = 0;
  for (auto x : *(opi->bond_ptr())) {
    state_[i] = state[x];
    i++;
  }
  ASSERT(is_same_state(local_state, state_, 0),
         "the operator can not be applied to the state");
#endif
  if (opi->is_off_diagonal()) {
    update_state_OD(opi, state);
  }
}

/*
 *update given state by given offdiagonal operator ptr;
 */
template <class MCT>
void Worm<MCT>::update_state_OD(typename OPS::iterator opi, state_t &state) {
  if (opi->_check_is_bond()) {
    int index = 0;
    auto const &bond = *(opi->bond_ptr());
    for (auto x : bond) {
      state[x] = opi->get_local_state(bond.size() + index);
      index++;
    }
  } else {
    int site = static_cast<int>(
        opi->bond_ptr() -
        &nn_sites[0]);  // calculate site from the difference of pointer values.
                        // (nn_sites.begin() is the first element of vector)
#ifndef NDEBUG
    if (state.at(site) != opi->get_local_state(0))
      throw std::runtime_error("state is not consistent");
#endif
    state[site] = opi->get_local_state(1);
  }
}

/*
* check the operator and state is consistent during the worm_updateg
params
------
worm_label : label of dot (Worm) we are aiming at.
p_label : label of dot before reaching at the current position of Worm.

*/
template <class MCT>
void Worm<MCT>::checkOpsInUpdate(int worm_label, int p_label, int t_dir,
                                 int t_fl, int fl, int dir) {
#ifndef NDEBUG
  auto state_ = state;

  int label = 0;
  d_cnt++;
  for (const auto &dot : spacetime_dots) {
    if (dot.at_operator() && (dot.index() == 0)) {  // if dot_type is operator
      auto opi = ops_main.begin() + dot.label();
      update_state(opi, state_);
    } else if (dot.at_origin()) {
      int dot_spin = state[dot.label()];
      ASSERT(state_[dot.site()] == dot_spin, "spin is not consistent");
    }
    label++;
  }
  ASSERT(is_same_state(state_, state, 0),
         "operators are not consistent while update worms");
#endif
  return;
}

template <class MCT>
bool Worm<MCT>::detectWormCross(double tau, double tau_prime, double wt_tau,
                                int dir) {
  if (dir == 1) {
    if ((tau_prime == 0 ? 1 : tau_prime) >= wt_tau && wt_tau > tau)
      return true;
    else
      return false;
  } else {  // if dir == 0
    if (tau_prime <= wt_tau && wt_tau < (tau == 0 ? 1 : tau))
      return true;
    else
      return false;
  }
}

template <class MCT>
void Worm<MCT>::reset_ops() {
  for (size_t i = 0; i < psop.size(); i++) {
    ops_main[psop[i]] = pops_main[i];
  }
}

template <class MCT>
double Worm<MCT>::get_single_flip_elem(const OP_type &op) {
  if (op._check_is_bond())
    throw std::runtime_error("op must be a single site operator");
  ptrdiff_t _index = op.bond_ptr() - &nn_sites[0];
  int site = static_cast<int>(_index);
  int x = op.get_local_state(0);
  int x_prime = op.get_local_state(1);
  const state_t &nn_state = op.nn_state();
  double mat_elem = 0;
  for (int i = 0; i < nn_sites[site].size(); i++) {
    auto target = spin_model.nn_sites[site][i];
    mat_elem += loperators[target.bt].single_flip(target.start, nn_state[i], x,
                                                  x_prime);
  }
  // if (x != x_prime) {
  //   int a = 0;
  //   vector<int> tmp_state(4);
  //   vector<int> tmp_state2(4);
  //   tmp_state[site] = x;
  //   tmp_state2[site] = x_prime;
  //   for (int j = 0; j < nn_sites[site].size(); j++) {
  //     tmp_state[nn_sites[site][j]] = nn_state[j];
  //     tmp_state2[nn_sites[site][j]] = nn_state[j];
  //   }
  //   dout << "site : " << site << "nn_sites : " << nn_sites[site] << endl;
  //   dout << "state = " << tmp_state << endl;
  //   dout << "state2 = " << tmp_state2 << endl;
  //   dout << "x : " << x << " x_prime : " << x_prime << endl;
  //   dout << "mat_elem : " << mat_elem << endl;
  //   int xxx = 0;
  // }
  return mat_elem;
}

template <class MCT>
double Worm<MCT>::get_single_flip_elem(int site, int x, int x_prime,
                                       state_t _state) {
  double mat_elem = 0;

  for (auto target : spin_model.nn_sites[site]) {
    mat_elem += loperators[target.bt].single_flip(
        target.start, _state[target.target], x, x_prime);
  }
  return mat_elem;
}

template <class MCT>
double Worm<MCT>::get_single_flip_elem(int site, int x, int x_prime,
                                       state_t _state, state_t &nn_state) {
  double mat_elem = 0;
  nn_state.resize(spin_model.nn_sites[site].size());

  for (int i = 0; i < spin_model.nn_sites[site].size(); i++) {
    auto target = spin_model.nn_sites[site][i];
    nn_state.at(i) = _state[target.target];
    mat_elem += loperators[target.bt].single_flip(
        target.start, _state[target.target], x, x_prime);
  }
  return mat_elem;
}

/*
params
------
dir : 0 or 1, 0 comes from bottom, 1 comes from top (moving direction is
reverse)

return
------
dir : 0 or 1, 0 exit from bottom, 1 comes from top (moving direction is same)
fl : flip site
*/
template <class MCT>
std::pair<int, int> Worm<MCT>::markov_next_flip(OP_type &op, int dir, int fl,
                                                bool zero_fl) {
  // d* error check
  if (dir != 0 && dir != 1) throw std::runtime_error("dir must be 0 or 1");
  if (op._check_is_bond())
    throw std::runtime_error("op must be a single site operator");
  if (fl < 0 || fl >= sps) throw std::runtime_error("fl must be in [0, sps)");

  int num = op.state();
  double mat_elem = get_single_flip_elem(op);
  double old_sign = sign;
  sign *= mat_elem > 0 ? 1 : -1;
  mat_elem = std::abs(mat_elem);
  if (fl == 0) {
    int x = 0;
  }
  op.update_state(dir, fl);
  double r = uniform(rand_src);

  int fl_prime, dir_prime;
  if (zero_fl) {
    int flip_prime = static_cast<int>((2 * sps - 1) * uniform(rand_src));
    if (flip_prime == 0) {
      dir_prime = 0;
      fl_prime = 0;
    } else {
      flip_prime--;
      dir_prime = flip_prime / (sps - 1);
      fl_prime = flip_prime % (sps - 1) + 1;
    }
    if (fl_prime < 0 || fl_prime >= sps)
      throw std::runtime_error("fl_prime must be in [0, sps)");
  } else {
    int flip_prime = static_cast<int>((2 * (sps - 1)) * uniform(rand_src));
    dir_prime = flip_prime / (sps - 1);
    fl_prime = flip_prime % (sps - 1) + 1;
    if (fl_prime < 1 || fl_prime >= sps)
      throw std::runtime_error("fl_prime must be in [0, sps)");
  }
  if (dir_prime != 0 && dir_prime != 1)
    throw std::runtime_error("dir_prime must be 0 or 1");

  // n* update op state using dir_prime and fl_prime
  op.update_state(dir_prime, fl_prime);
  double mat_elem_prime = get_single_flip_elem(op);
  sign *= mat_elem_prime > 0 ? 1 : -1;
  mat_elem_prime = std::abs(mat_elem_prime);

  //* update op state using Metropolis
  double ratio = mat_elem_prime / mat_elem;

  if (r < ratio)
    return make_pair(dir_prime, fl_prime);
  else {
    op.set_state(num);
    sign = old_sign;
    return make_pair(dir, (sps - fl) % sps);
  }
}

/*
dir_in is direction from operator's perspective.
----------------------------
pair<dir_out, fl>
*/
template <class MCT>
std::pair<int, int> Worm<MCT>::markov_diagonal_nn(OP_type &op, int dir_in,
                                                  int fl, int nn_index) {
  if (dir_in != 0 && dir_in != 1)
    throw std::runtime_error("dir must be 0 or 1");
  if (op._check_is_bond())
    throw std::runtime_error("op must be a single site operator");
  if (fl < 0 || fl >= sps) throw std::runtime_error("fl must be in [0, sps)");

  int num = op.state();
  // double mat_elem = std::abs(get_single_flip_elem(op));
  double old_sign = sign;
  double mat_elem = get_single_flip_elem(op);
  sign *= mat_elem > 0 ? 1 : -1;
  mat_elem = std::abs(mat_elem);
  op.update_nn_state(nn_index, fl);

  // double mat_elem_prime = std::abs(get_single_flip_elem(op));
  double mat_elem_prime = get_single_flip_elem(op);
  sign *= mat_elem_prime > 0 ? 1 : -1;
  mat_elem_prime = std::abs(mat_elem_prime);
  double ratio = mat_elem_prime / mat_elem;
  double r = uniform(rand_src);

  if (r < ratio)
    return make_pair(!dir_in, fl);
  else {
    op.update_nn_state(nn_index, (sps - fl) % sps);
    sign = old_sign;
    return make_pair(dir_in, (sps - fl) % sps);
  }
}

template <class MCT>
void Worm<MCT>::printStateAtTime(const state_t &state, double time) {
#ifndef NDEBUG
  dout << "current spin at time :" << time << " is : ";
  for (auto spin : state) {
    dout << spin << " ";
  }
  dout << std::endl;
#endif
  return;
}

template class Worm<bcl::heatbath>;
template class Worm<bcl::st2010>;
template class Worm<bcl::st2013>;
