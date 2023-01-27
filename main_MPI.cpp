#include <iostream>
#include <string>
#include <fstream>
#include <functional>
#include <libconfig.h++>
#include <dirent.h>
#include <filesystem>
#include <unistd.h>
#include <automodel.hpp>
#include <autoobservable.hpp>
#include <exec_parallel.hpp>
#include <options.hpp>
#include <argparse.hpp>
#include <observable.hpp>
#include <funcs.hpp>
#include <mpi.h>

#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include <jackknife.hpp>
#include <alps/alea/batch.hpp>
#include <alps/utilities/mpi.hpp>


using namespace std;
using namespace libconfig;

using namespace std;

double elapsed;


int main(int argc, char **argv) {
  
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //set up alps::mpi::reducer
  alps::alea::mpi_reducer red_(alps::mpi::communicator(), 0);
  alps::alea::reducer_setup setup = red_.get_setup();

  // alps::mpi::communicator comm_;

  // cout << rank << endl;

  char tmp[256];
  auto _ = getcwd(tmp, 256);
  // cout << tmp << endl;

  Config cfg;
  cfg.setAutoConvert(true);

  //* argparse  
  argparse::ArgumentParser parser("test", "argparse test program", "Apache License 2.0");

  parser.addArgument({"-L1"}, "set shape[0]");
  parser.addArgument({"-L2"}, "set shape[1]");
  parser.addArgument({"-L3"}, "set shape[2]");
  parser.addArgument({"-N"}, "# of montecarlo steps (sweeps)");
  parser.addArgument({"-K"}, "# of montecarlo steps for thermalization");
  parser.addArgument({"--split-sweeps"},
              "bool that determines whether to split # sweeps among processes",
              argparse::ArgumentType::StoreTrue
              );

  parser.addArgument({"-T"}, "set temperature");
  parser.addArgument({"-m"}, "model name");
  parser.addArgument({"-ham"}, "path to hamiltonian");
  parser.addArgument({"-obs"}, "path to observables");
  parser.addArgument({"-P1"}, "set params[0]");
  parser.addArgument({"-P2"}, "set params[1]");

  auto args = parser.parseArgs(argc, argv);


  try { cfg.readFile("/home/user/project/config/model.cfg");}
  catch(const FileIOException &fioex)
  {
    cerr << "I/O error while reading file." << endl;
    return(EXIT_FAILURE);
  }
  catch(const ParseException &px)
  {
    cerr << "error while parsing items" << endl;
    cerr << "Maybe some list include multiple types (e.g. = [1.0, 1, 1])" << endl;
    return(EXIT_FAILURE);
  }

  const Setting& root = cfg.getRoot();
  string model_name = root["model"];
  bool print_lat = (bool) root["print_lattice"];
  model_name = args.safeGet<string>("m", model_name);
  if(rank == 0) {
    cout << "model name is \t : \t" << model_name << endl;
    cout << "run on \t : \t" << size << " nodes" << endl;
  }
  const Setting& mcfg = root["models"][model_name];
  const Setting& shape_cfg = mcfg.lookup("length");
  const Setting& params_cfg = mcfg.lookup("params");
  const Setting& types_cfg = mcfg.lookup("types");
  const Setting& dofs_cfg = mcfg.lookup("dofs");


  double shift;
  string file, basis, cell, ham_path, obs_path;
  bool repeat; // true if repeat params and types.
  bool zero_worm;
  vector<size_t> shapes;
  vector<int> types;
  vector<double> params;
  vector<size_t> dofs;

  for (int i=0; i<shape_cfg.getLength(); i++) {int tmp = shape_cfg[i]; shapes.push_back(tmp);}
  for (int i=0; i<dofs_cfg.getLength(); i++) {dofs.push_back((size_t)dofs_cfg[i]);}
  for (int i=0; i<params_cfg.getLength(); i++) {params.push_back((float)params_cfg[i]);}
  for (int i=0; i<types_cfg.getLength(); i++) {types.push_back(types_cfg[i]);}


  file = (string) mcfg.lookup("file").c_str();
  basis = (string) mcfg.lookup("basis").c_str();
  cell = (string) mcfg.lookup("cell").c_str();
  ham_path = (string) mcfg.lookup("ham_path").c_str();
  try { obs_path = (string) mcfg.lookup("obs_path").c_str();}
  catch(const SettingNotFoundException &nfex) { obs_path = "";}
  repeat = (bool) mcfg.lookup("repeat");
  shift = (double) mcfg.lookup("shift");
  zero_worm = (bool) mcfg.lookup("zero_worm");

  // cout << file << endl;

  //* settings for monte-carlo
  const Setting& settings = root["mc_settings"];

  size_t sweeps, therms, cutoff_l;
  double T = 0;
  bool fix_wdensity = false;
  try
  {
    const Setting& config = settings["config"];
    sweeps = (long) config.lookup("sweeps");
    therms = (long) config.lookup("therms");
    cutoff_l = (long) config.lookup("cutoff_length");
    T = (double) config.lookup("temperature");
    fix_wdensity = config.lookup("fix_wdensity");

  }
  catch(...)
  {
    cout << "I/O error while reading mc_settings.default settings" << endl;
    cout << "read config file from default instead" << endl;
    const Setting& config = settings["default"];
    sweeps = (long) config.lookup("sweeps");
    therms = (long) config.lookup("therms");
    cutoff_l = (long) config.lookup("cutoff_length");
    T = (double) config.lookup("temperature");
    fix_wdensity = config.lookup("fix_wdensity");
  }



  // parser
  shapes[0] = args.safeGet<size_t>("L1", shapes[0]);
  shapes[1] = args.safeGet<size_t>("L2", shapes[1]);
  shapes[2] = args.safeGet<size_t>("L3", shapes[2]);
  T = args.safeGet<double>("T", T);
  sweeps = args.safeGet<int>("N", sweeps);
  therms = args.safeGet<int>("K", therms);
  params[0] = args.safeGet<float>("P1",  params[0]);
  params[1] = args.safeGet<float>("P2",  params[1]);

  try { 
    ham_path = args.get<string>("ham");
    try { obs_path = args.get<string>("obs");}
    catch(...) { 
      if (rank == 0) cout << "obs_path is not given. Elements of observables are set to zero" << endl;
      obs_path = "";
    }
  }
  catch(...) {}

  if (args.has("split-sweeps")) sweeps = sweeps / size;
  sweeps = (sweeps / 2) * 2; // make sure sweeps is even number
  if (rank == 0){
    cout << "zero_wom : " << (zero_worm ? "YES" : "NO") << endl;
    cout << "repeat : " << (repeat ? "YES" : "NO") << endl;
    cout << "params : " << params << endl;
  }


  //* finish argparse


  model::base_lattice lat(basis, cell, shapes, file, !rank);
  model::base_model<bcl::st2013> spin(lat, dofs, ham_path, params, types, shift, zero_worm, repeat, !rank);
  model::observable obs(spin, obs_path, !rank);

  // output MC step info 
  if (rank == 0 ) cout << "therms(each process)    : " << therms << endl
                       << "sweeps(each process)    : " << sweeps << endl 
                       << "sweeps(in total)        : " << sweeps * size << endl;

  if (rank == 0) {for (int i=0; i<40; i++) cout << "-"; cout << endl;}



  // simulate with worm algorithm (parallel computing is enable)
  vector<batch_res> res;
  exe_worm_parallel(spin, T, sweeps, therms, cutoff_l, fix_wdensity, rank, res, obs);  


  batch_res as = res[0]; // average sign 
  batch_res ene = res[1]; // signed energy i.e. $\sum_i E_i S_i / N_MC$
  batch_res sglt = res[2];
  batch_res n_neg_ele = res[3];
  batch_res n_ops = res[4];
  batch_res N2 = res[5];
  batch_res N = res[6];
  batch_res dH = res[7]; // $\frac{\frac{\partial}{\partial h}Z}{Z}$ 
  batch_res dH2 = res[8]; // $\frac{\frac{\partial^2}{\partial h^2}Z}{Z}$


  as.reduce(red_);
  ene.reduce(red_);
  sglt.reduce(red_);
  n_neg_ele.reduce(red_);
  n_ops.reduce(red_);
  N2.reduce(red_);
  N.reduce(red_);
  dH.reduce(red_);
  dH2.reduce(red_);

  double elapsed_max, elapsed_min;
  MPI_Allreduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&elapsed, &elapsed_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);


  if (setup.have_result) {
    
    std::function<double(double, double, double)> f;

    pair<double, double> as_mean = jackknife_reweight_single(as);  // calculate <S>
    pair<double, double> nop_mean = jackknife_reweight_single(n_ops);  // calculate <S>
    pair<double, double> nnop_mean = jackknife_reweight_single(n_neg_ele);  // calculate <S>



    // calculate energy
    pair<double, double> ene_mean = jackknife_reweight_div(ene, as);  // calculate <SH> / <S>


    // calculat heat capacity
    f = [](double x1, double x2, double y) { return (x2 - x1)/y - (x1/y)*(x1/y); }; 
    pair<double, double > c_mean = jackknife_reweight_any(N, N2, as, f);  

    // calculate magnetization
    pair<double, double> m_mean = jackknife_reweight_div(dH, as); 

    // calculate susceptibility
    f = [](double x1, double x2, double y) { return x2/y - (x1/y)*(x1/y); };
    pair<double, double> chi_mean = jackknife_reweight_any(dH, dH2, as, f); 


    cout << "Elapsed time         = " << elapsed_max << "(" << elapsed_min <<") sec\n"
         << "Speed                = " << (therms+sweeps) / elapsed_max << " MCS/sec\n";

    cout << "beta                 = " << 1.0 / T 
         << endl
         << "Total Energy         = "
         << ene_mean.first << " +- " 
         << ene_mean.second
         << endl;
    
    cout << "Average sign         = "
         << as_mean.first << " +- " 
         << as_mean.second 
         << endl
         << "Energy per site      = "
         << ene_mean.first / lat.L << " +- " 
         << ene_mean.second / lat.L
         << endl
         << "Specific heat        = "
         << c_mean.first / lat.L << " +- " 
         << c_mean.second / lat.L
         << endl
         << "magnetization        = "
         << m_mean.first * T / lat.L << " +- " << m_mean.second * T / lat.L
         << endl
         << "susceptibility       = "
         << chi_mean.first  * T / lat.L << " +- " << chi_mean.second  * T / lat.L << endl
         << "# of operators       = "
         << nop_mean.first << " +- " << nop_mean.second << endl
         << "# of neg sign op     = "
         << nnop_mean.first << " +- " << nnop_mean.second << endl;

  }
  MPI_Finalize();

}