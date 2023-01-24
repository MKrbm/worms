#include <iostream>
#include <string>
#include <fstream>
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


using namespace std;
using namespace libconfig;

using namespace std;

template <class T>
class VectorPlus {
public:
  T operator()(const T &lhs, const T &rhs) const {
    const auto n = lhs.size();
    auto r = T(n);
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), r.begin(), std::plus<>());
    return r;
  }
};


int main(int argc, char **argv) {

  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  boost::mpi::communicator world;

  // cout << world.rank() << endl;

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
  parser.addArgument({"-N"}, "# of montecarlo steps");
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
  model_name = args.safeGet<std::string>("m", model_name);
  if(world.rank() == 0) {
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


  // boost::serialization::access world;
  BC::observable local_result;

  // parser
  shapes[0] = args.safeGet<size_t>("L1", shapes[0]);
  shapes[1] = args.safeGet<size_t>("L2", shapes[1]);
  shapes[2] = args.safeGet<size_t>("L3", shapes[2]);
  T = args.safeGet<double>("T", T);
  sweeps = args.safeGet<int>("N", sweeps);
  params[0] = args.safeGet<float>("P1",  params[0]);
  params[1] = args.safeGet<float>("P2",  params[1]);

  try { 
    ham_path = args.get<std::string>("ham");
    try { obs_path = args.get<std::string>("obs");}
    catch(...) { 
      if (rank == 0) cout << "obs_path is not given. Elements of observables are set to zero" << endl;
      obs_path = "";
    }
  }
  catch(...) {}

  sweeps = sweeps / size;
  if (rank == 0){
    cout << "zero_wom : " << (zero_worm ? "YES" : "NO") << endl;
    cout << "repeat : " << (repeat ? "YES" : "NO") << endl;
    cout << "params : " << params << endl;
  }


  //* finish argparse


  model::base_lattice lat(basis, cell, shapes, file, !world.rank());
  model::base_model<bcl::st2013> spin(lat, dofs, ham_path, params, types, shift, zero_worm, repeat, !world.rank());
  model::observable obs(spin, obs_path, !world.rank());

  // output MC step info 
  if (rank == 0 ) cout << "therms(each process)    : " << therms << endl
                       << "sweeps(each process)    : " << sweeps << endl 
                       << "sweeps(in total)        : " << sweeps * size << endl;

  if (rank == 0) {for (int i=0; i<40; i++) cout << "-"; cout << endl;}



  // simulate with worm algorithm (parallel computing is enable)
  std::vector<BC::observable> res;
  exe_worm_parallel(spin, T, sweeps, therms, cutoff_l, fix_wdensity, rank, res, obs);  


  auto _res = boost::mpi::all_reduce(world, res, VectorPlus<std::vector<BC::observable>>()); //all reduce (sum over all results)


  if (world.rank()==0)
  {
    BC::observable ene=_res[0]; // signed energy i.e. $\sum_i E_i S_i / N_MC$
    BC::observable ave_sign=_res[1]; // average sign 
    BC::observable sglt=_res[2]; 
    BC::observable n_neg_ele=_res[3]; 
    BC::observable n_ops=_res[4]; 
    BC::observable N2 =_res[5];
    BC::observable N =_res[6];
    BC::observable dH =_res[7]; // $\frac{\frac{\partial}{\partial h}Z}{Z}$ 
    BC::observable dH2 =_res[8]; // $\frac{\frac{\partial^2}{\partial h^2}Z}{Z}$

    double ene_err = std::sqrt(std::pow(ene.error()/ave_sign.mean(), 2) + std::pow(ene.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2));
    double ene_mean = ene.mean()/ave_sign.mean();

    double c_mean = (N2.mean() - N.mean()) / ave_sign.mean() - (N.mean() / ave_sign.mean()) * (N.mean() / ave_sign.mean());
    double c_err = std::sqrt(std::pow(N2.error()/ave_sign.mean(), 2) + std::pow(N.error()/ave_sign.mean(), 2) + std::pow(N2.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2) + std::pow(N.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2));
    double n_err = std::sqrt(std::pow(N.error()/ave_sign.mean(), 2) + std::pow(N.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2));
    c_err += 2 * n_err;

    double _dH2 = dH2.mean() / ave_sign.mean();
    double _dH = dH.mean() / ave_sign.mean();
    double mag = _dH * T / lat.L ;
    double mag_err = std::sqrt(std::pow(dH.error()/ave_sign.mean(), 2) + std::pow(dH.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2)) * T / lat.L;
    double sus = (_dH2 - _dH * _dH ) * T / lat.L; // suscetibility
    double sus_err = std::sqrt(std::pow(dH2.error()/ave_sign.mean(), 2) + std::pow(dH2.mean()/std::pow(ave_sign.mean(),2) * ave_sign.error(),2)) * T / lat.L;
    sus_err += 2 * _dH * mag_err;
    

    
    std::cout << "beta                 = " << 1.0 / T << endl;
    std::cout << "Total Energy         = "
            << ene.mean()/ave_sign.mean()<< " +- " 
            << ene_err
            << std::endl;

    // std::cout << "Elapsed time         = " << elapsed << " sec\n"
    //           << "Speed                = " << (therms+sweeps) / elapsed << " MCS/sec\n";
    std::cout << "Energy per site      = "
              << ene.mean()/ave_sign.mean() / lat.L << " +- " 
              << ene_err / lat.L
              << std::endl
              << "specific heat        = "
              << c_mean / lat.L << " +- " 
              << c_err / lat.L
              << std::endl
              << "magnetization        = "
              << mag << " +- " << mag_err << std::endl 
              << "susceptibility       = "
              << sus << " +- " << sus_err << std::endl
              << "average sign         = "
              << ave_sign.mean() << " +- " << ave_sign.error() << std::endl
              << "dimer operator       = "
              << sglt.mean() << std::endl 
              << "# of operators       = "
              << n_ops.mean() << std::endl
              << "# of neg sign op     = "
              << n_neg_ele.mean() << std::endl;
  }

  MPI_Finalize();

}