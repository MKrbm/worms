#include "../include/exec_parallel.hpp"

// parallel version of exe_worm
template <typename MC>
std::unordered_map<std::string, model::WormObs> exe_worm_parallel(
    model::base_model<MC> spin_model,
    double T,
    size_t sweeps,
    size_t therms,
    size_t cutoff_l,
    bool fix_wdensity,
    int rank,
    std::vector<batch_res> &res, // contains results such as energy, average_sign,, etc
    alps::alea::autocorr_result<double> &ac_res,
    model::observable obs,
    model::MapWormObs wobs)
{

  // cout << "Hi" << endl;
  using SPINMODEL = model::base_model<MC>;
  if (cutoff_l < 0)
    cutoff_l = numeric_limits<decltype(cutoff_l)>::max();

  batch_obs ave_sign(1); // average sign
  batch_obs ene(1);      // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_obs n_neg_ele(1);
  batch_obs n_ops(1);
  batch_obs N2(1);                               // average of square of number of operators (required for specific heat)
  batch_obs N(1);                                // average of number of operators (required for specific heat)
  batch_obs dH2(1);                              // second derivative by magnetic field
  batch_obs dH(1);                               // first derivative by magnetic field
  batch_obs m_diag(1);                           // magnetization (assume standard basis)
  alps::alea::autocorr_acc<double> auto_corr(3); // H, m^2, S

  // ; // magnetization
  BC::observable M2; // magnetization^2
  BC::observable K;  // matnetic susceptibility

  double beta = 1 / T;

  Worm<MC> solver(beta, spin_model, wobs, cutoff_l, rank); // template needs for std=14
  // spin_model.lattice.print(std::cout);

#if MESTIME
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
// double du_time = 0;
// double wu_time = 0;
#endif

  int n_kink = 0;
  int cnt = 0;
  solver.initStates();
  double wcount = 0;
  double wlength = 0;
  double wdensity = spin_model.Nb;
  for (int i = 0; i < therms + sweeps; i++)
  {
    solver.diagonalUpdate(wdensity); // n* need to be comment out
    solver.wormUpdate(wcount, wlength);
    if (cnt >= therms)
    {
      int sign = 1;
      // double w_rate = 1;
      double n_neg = 0;
      double n_op = 0;
      double mu = 0;
      double sum_ot = 0;   // \sum_{tau} O_{tau} : sum of observables
      double sum_2_ot = 0; // \sum_{tau} (O_{tau})^2 : sum of square of observables
      for (const auto &s : solver.state)
      {
        mu += 0.5 - s;
      }
      for (const auto &op : solver.ops_main)
      {
        int sign_ = 1;
        if (op._check_is_bond()){
          sign_ = spin_model.loperators[op.op_type()].signs[op.state()];
        } else {
          sign_ = solver.get_single_flip_elem(op)  > 0 ? 1 : -1;
        }
        sign *= sign_;
        if (sign_ == -1)
          n_neg++;
        n_op++;

        // calculate kai (susceptibility)
        double _op_tmp = obs.obs_operators(op.op_type(), op.state());
        // if (_op_tmp != 0) {
        //   cout << op.get_state_vec() << " " << _op_tmp << " " << _op_tmp << endl;
        // }
        sum_ot += _op_tmp;
        sum_2_ot += _op_tmp * _op_tmp;
      }
      // cout << solver.sign << sign << endl;
      if (solver.sign != sign)
      {
        cout << "sign is not consistent" << endl;
        exit(1);
      }
      double m = (double)solver.ops_main.size();
      double ene_tmp = -m / beta + spin_model.shift();
      N2 << (m * m) * sign;
      N << m * sign;
      ene << ene_tmp * sign;
      ave_sign << sign;
      n_neg_ele << n_neg;
      n_ops << n_op;
      dH << sum_ot * sign;
      dH2 << (sum_ot * sum_ot - sum_2_ot) * sign;
      mu /= spin_model.L;
      m_diag << (mu * sign);
      // alps::alea::vector_adapter<double> _tmp = std::vector<double>({ene_tmp, mu * mu, (double)sign});
      auto v = std::vector<double>({ene_tmp, mu * mu, (double)sign});
      auto _tmp = alps::alea::make_adapter<double>(v);
      // cout << alps::alea::make_adapter<double>(v)[0] << endl;
      auto_corr << _tmp;
    }
    if (i <= therms / 2)
    {
      if (!fix_wdensity)
      {
        if (wcount > 0)
          wdensity = spin_model.Nb / (wlength / wcount); // Actually at least one worm is secured.
        if (i % (therms / 8 + 1) == 0)
        {
          wcount /= 2;
          wlength /= 2;
        }
      }
    }
    if (i == therms / 2)
    {
      if (!fix_wdensity && (rank == 0))
        std::cout << "Info: average number worms per MCS is reset from " << spin_model.L
                  << " to " << wdensity + 1 << "(rank=" << rank << ")"
                  << "\n\n";
      else if (rank == 0)
        std::cout << "Info: average number worms per MCS is " << wdensity + 1 << "(rank=" << rank << ")"
                  << "\n\n";
    }
    cnt++;
  }

#if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
#endif
  double r = 1 - solver.bocnt / (double)(therms + sweeps); // # of loops breaked out divded by total number of loops.
  // double r_ = 1-r_;

  res.push_back(ave_sign.finalize());
  res.push_back(ene.finalize());
  res.push_back(n_neg_ele.finalize());
  res.push_back(n_ops.finalize());
  res.push_back(N2.finalize());
  res.push_back(N.finalize());
  res.push_back(dH.finalize());
  res.push_back(dH2.finalize());
  res.push_back(solver.get_phys_cnt().finalize());
  for (auto &obs : solver.get_worm_obs())
  {
    res.push_back(obs.second.finalize());
  }

  ac_res = auto_corr.finalize();
  return solver.get_worm_obs();
}

template map_wobs_t exe_worm_parallel<bcl::st2013>(
    model::base_model<bcl::st2013> spin_model,
    double T,
    size_t sweeps,
    size_t therms,
    size_t cutoff_l,
    bool fix_wdensity,
    int rank,
    std::vector<batch_res> &res,
    alps::alea::autocorr_result<double> &ac_res,
    model::observable obs,
    model::MapWormObs wobs);

template map_wobs_t exe_worm_parallel<bcl::st2010>(
    model::base_model<bcl::st2010> spin_model,
    double T,
    size_t sweeps,
    size_t therms,
    size_t cutoff_l,
    bool fix_wdensity,
    int rank,
    std::vector<batch_res> &res,
    alps::alea::autocorr_result<double> &ac_res,
    model::observable obs,
    model::MapWormObs wobs);

template map_wobs_t exe_worm_parallel<bcl::heatbath>(
    model::base_model<bcl::heatbath> spin_model,
    double T,
    size_t sweeps,
    size_t therms,
    size_t cutoff_l,
    bool fix_wdensity,
    int rank,
    std::vector<batch_res> &res,
    alps::alea::autocorr_result<double> &ac_res,
    model::observable obs,
    model::MapWormObs wobs);
