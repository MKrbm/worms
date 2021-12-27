// #define RANDOM_SEED 0
#include "MainConfig.h"

#include <iostream>
#include <model.hpp>
#include <state.hpp>
#include <string>
#include <chrono>
#include <observable.hpp>

#define DEBUG 1
#define MCSTEP 1E5
#define SWEEP 1E4
#define MESTIME 1

#if MESTIME
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  using std::chrono::microseconds;
  using MODEL = model::heisenberg1D;
  using OPS = std::vector<spin_state::Operatorv2>;
  using STATE = std::vector<int>;
  using WORMS = spin_state::WORM_ARR;
  using DOTS = std::vector<spin_state::Dotv2>;
  using spin_state_t = spin_state::spin_state<2, 2>;

#endif

//*append to ops
void append_ops(OPS& ops, std::vector<int> const& bond,  int state, int op_type, double tau){
  ops.emplace_back(bond, state, bond.size(), op_type, tau);
}
//*overload for r value
void append_ops(OPS& ops, std::vector<int> && bond,  int state, int op_type, double tau){
  ops.emplace_back(bond, state, bond.size(), op_type, tau);
}

//*append to worms
void append_worms(WORMS& wm, int site, int spin, int dot_label, double tau){
  wm.emplace_back(site, spin, dot_label, tau);
}


void set_dots(int site, int dot_type, int index, DOTS& spt, OPS& ops, WORMS& worms_list){
  int label = spt.size();

  if (dot_type == -1) {
    ASSERT(label == site, "label must be equal to site");
    spt.push_back(
      spin_state::Dotv2::state(site)
    );
  }else if(dot_type == 0){
    int n = ops.size();
    spt.emplace_back(
      site, spt[site].prev(), site, n-1,index
    );
    spt[spt[site].prev()].set_next(label);
    spt[site].set_prev(label);
  }else if(dot_type == -2){
    int n = worms_list.size();
    spt.push_back(
      spin_state::Dotv2::worm(site, spt[site].prev(), site, n-1)
    );
    spt[spt[site].prev()].set_next(label);
    spt[site].set_prev(label);
  }
}
int main(int argc, char* argv[])
{
  if (argc < 4) {
    // report version
    std::cout << argv[0] << " Version " << VERSION_MAJOR << "."
              << VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " L J beta" << std::endl;
    return 1;
  }
  std::cout << "MC step : " << MCSTEP << "\n" 
            << "sweep size : " << SWEEP << std::endl;

  int L = std::stoi(argv[1]);
  double J = std::stoi(argv[2]);
  double beta = std::stoi(argv[3]);
  double h = 0;
  BC::observable ene; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  BC::observable umag; // uniform magnetization 
  BC::observable ave_sign; // average sign 




  model::heisenberg1D h1(L,h,J);



  OPS ops; //contains operators.
  OPS ops_sub; // for sub.
  STATE state(L);
  STATE cstate(L);
  DOTS spt; //contain dots in space-time.
  WORMS worms_list;

  std::vector< std::vector<int> > bonds(h1.bonds);

  typedef std::mt19937 engine_type;
  #ifdef RANDOM_SEED
  engine_type rand_src = engine_type(static_cast <unsigned> (time(0)));
  #else
  engine_type rand_src = engine_type(2021);
  #endif


  // random distribution from 0 to 1
  typedef std::uniform_real_distribution<> uniform_t;
  typedef std::exponential_distribution<> expdist_t;
  uniform_t uniform;
  static const int N_op = MODEL::Nop;

  std::array<model::local_operator, N_op>& loperators(h1.loperators); //holds multiple local operators
  std::array<int, N_op>& leg_sizes(h1.leg_size); //leg size of local operators;
  std::array<double, N_op>& operator_cum_weights(h1.operator_cum_weights);



  #if MESTIME
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  double du_time = 0;
  double wu_time = 0;
  #endif


  int n_kink=0;
  int cnt = 0;
  int spin = 1;
  for (auto& s : state){
    s = spin;
    spin^=1;
  }

  int wdensity = 3;
  for (int i=0; i < MCSTEP + SWEEP; i++){
    std::swap(ops, ops_sub);
    expdist_t expdist(h1.rho * beta + wdensity); //initialize exponential distribution
    double pstart = wdensity / (beta * h1.rho + wdensity); //probability of choosing worms
    std::copy(state.begin(), state.end(), cstate.begin());
    std::size_t lop_label;
    lop_label = 0; //typically, lop_label is fixed to 0
    int leg_size = leg_sizes[lop_label]; //size of choosen operator
    auto& lop = loperators[lop_label];

    ops.resize(0); //* init_ops()
    
    spt.resize(0);
    for(int i=0; i<L; i++){
      set_dots(i, -1, i, spt, ops, worms_list);
    }

    //*init worms
    worms_list.resize(0);
    ops_sub.push_back(spin_state::Operatorv2::sentinel(1)); //*sentinels
    double tau = expdist(rand_src);
    for (OPS::iterator opi = ops_sub.begin(); opi != ops_sub.end();){
      // auto op_sub = *opi;
      if (tau < opi->tau()){ //* if new point is behind the next operator is opsub.
        double r = uniform(rand_src);

        if (r < pstart){
          int s = static_cast<int>(h1.L * uniform(rand_src));
          append_worms(worms_list, s, cstate[s],spt.size(), tau);
          set_dots(s, -2 , 0, spt, ops, worms_list); //*index is always 0 
        }else{
          int b = static_cast<int>(bonds.size() * uniform(rand_src));
          const auto& bond = bonds[b];
          // int u = spin_state::state2num(cstate, bond);
          // int u = spin_state_t::c2u(cstate[bond[0]], cstate[bond[1]]);
          int s0 = bond[0];
          int s1 = bond[1];
          int u = spin_state_t::c2u(cstate[s0], cstate[s1]);
          r = uniform(rand_src);
          if (r < lop.accept[u]){
            append_ops(ops, bond, (u<<bond.size()) | u, lop_label, tau);
            // append_ops(ops, bond, spin_state_t::u2p(u, u), lop_label, tau);
            
            // for (int i=0; i<leg_size; i++){
            //   set_dots(bond[i], 0, i);
            // }
            set_dots(s0, 0, 0, spt, ops, worms_list);
            set_dots(s1, 0, 1, spt, ops, worms_list);

          }
        }
        tau += expdist(rand_src);
      }else{ //*if tau went over the operator time.
        // if (opi->is_off_diagonal()) {
        //   update_state(opi, cstate);
        //   ops.push_back(*opi);
        //   for (int i=0; i<opi->size(); i++){
        //     set_dots(opi->bond(i), 0 , i);
        //   }
        //   printStateAtTime(cstate, tau);
        // }
        ++opi;
      }
    } //end of while loop
    int xxx=0;

    if (cnt >= SWEEP){
      int sign = 1;
      double mu = 0;
      for (const auto&  s : state) {
        mu += 0.5 - s;
      }
      for (const auto& op : ops){
        // std::vector<int> local_state = *op;
        // int num = spin_state::state2num(local_state);
        // sign *= h1.loperators[op.op_type()];
        sign *= h1.loperators[op.op_type()].signs[op.state()];
        // sign *= op->plop->signs[num];
      }
      ene << (- ((double)ops.size()) / beta + h1.shifts[0] * h1.Nb) * sign;
      ave_sign << sign;
      mu /= h1.L;
      umag << mu * sign;
    }
    cnt++;
  }


  

  #if MESTIME
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  cout << "time for diagonal_update : " << du_time/(MCSTEP + SWEEP) << endl
            << "time for worm update : " << wu_time/(MCSTEP+SWEEP) << endl;

  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / (double)1E3;
  #endif
  std::cout << "Elapsed time = " << elapsed << " sec\n"
            << "Speed = " << (MCSTEP + SWEEP) / elapsed << " MCS/sec\n";
  std::cout << "Energy             = "
            << ene.mean()/ave_sign.mean() / h1.L << " +- " 
            << std::sqrt(std::pow(ene.error()/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2)) / h1.L
            << std::endl
            << "Uniform Magnetization     = "
            << umag.mean()/ave_sign.mean() << " +- " 
            << std::sqrt(std::pow(umag.error()/ave_sign.mean(), 2) + std::pow(umag.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2))
            << std::endl
            << "average sign     = "
            << ave_sign.mean() << " +- " << ave_sign.error() << std::endl;
}