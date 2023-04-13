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

// #include <boost/mpi.hpp>
// #include <boost/serialization/serialization.hpp>
// #include <boost/serialization/access.hpp>
// #include <boost/serialization/base_object.hpp>

#include <jackknife.hpp>
#include <alps/alea/batch.hpp>
#include <alps/utilities/mpi.hpp>




using namespace std;
using namespace libconfig;

using namespace std;



double elapsed;


int main(int argc, char **argv) {
  
  #ifdef WORKING_DIR
  chdir(WORKING_DIR);
  #endif

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
  parser.addArgument({"--z"}, "bool : introduce zero worm",
                argparse::ArgumentType::StoreTrue
                );

  parser.addArgument({"-T"}, "set temperature");
  parser.addArgument({"-m"}, "model name");
  parser.addArgument({"-ham"}, "path to hamiltonian");
  parser.addArgument({"-obs"}, "path to observables");
  parser.addArgument({"-wobs"}, "path to worm observables");
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
  const Setting *mcfg;
  try {mcfg = &root["models"][model_name];}
  catch(const SettingNotFoundException &nfex)
  {
    throw std::runtime_error("model name not found");
    return(EXIT_FAILURE);
  }
  // const Setting& mcfg = root["models"][model_name];
  const Setting& shape_cfg = mcfg->lookup("length");
  const Setting& params_cfg = mcfg->lookup("params");
  const Setting& types_cfg = mcfg->lookup("types");
  const Setting& dofs_cfg = mcfg->lookup("dofs");



  double shift;
  string file, basis, cell, ham_path, obs_path;
  vector<string> wobs_paths;
  bool repeat; // true if repeat params and types.
  bool zero_worm;
  size_t ns_unit;
  vector<size_t> shapes;
  vector<int> types;
  vector<double> params;
  vector<size_t> dofs;

  for (int i=0; i<shape_cfg.getLength(); i++) {int tmp = shape_cfg[i]; shapes.push_back(tmp);}
  for (int i=0; i<dofs_cfg.getLength(); i++) {dofs.push_back((size_t)dofs_cfg[i]);}
  for (int i=0; i<params_cfg.getLength(); i++) {params.push_back((float)params_cfg[i]);}
  for (int i=0; i<types_cfg.getLength(); i++) {types.push_back(types_cfg[i]);}


  file = (string) mcfg->lookup("file").c_str();
  basis = (string) mcfg->lookup("basis").c_str();
  cell = (string) mcfg->lookup("cell").c_str();
  ham_path = (string) mcfg->lookup("ham_path").c_str();
  try { obs_path = (string) mcfg->lookup("obs_path").c_str();}
  catch(const SettingNotFoundException &nfex) { obs_path = "";}
  repeat = (bool) mcfg->lookup("repeat");
  shift = (double) mcfg->lookup("shift");
  zero_worm = (bool) mcfg->lookup("zero_worm");
  try {ns_unit = (size_t) mcfg->lookup("ns_unit");} catch(...) {
    if (rank == 0) {
      cout << "Warning!!: please set ns_unit in model.cfg" << endl;
      cout << "ns_unit is automatically set to 1" << endl;
    }
    ns_unit = 1;
    }
  try { 
    const Setting& wobs_path_list = mcfg->lookup("worm_obs_path");
    for (int i=0; i<wobs_path_list.getLength(); i++) {
      string tmp = wobs_path_list[i]; 
      wobs_paths.push_back(tmp);
    }
  }
  catch(const SettingNotFoundException &nfex) {
    cout << "no worm observables" << endl;
  }

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
    try { wobs_paths = vector<string>(1,args.get<string>("wobs"));}
    catch(...) { 
      if (rank == 0) cout << "wobs_path is not given. Elements of worm observables will set to zero" << endl;
    }
  }
  catch(...) {}

  if (args.has("split-sweeps")) sweeps = sweeps / size;
  if (args.has("z")) zero_worm = true;

  sweeps = (sweeps / 2) * 2; // make sure sweeps is even number
  if (rank == 0){
    cout << "zero_wom : " << (zero_worm ? "YES" : "NO") << endl;
    cout << "repeat : " << (repeat ? "YES" : "NO") << endl;
    cout << "params : " << params << endl;
  }


  //* finish argparse

  model::base_lattice lat(basis, cell, shapes, file, !rank);
  model::base_model<bcl::st2013> spin(lat, dofs, ham_path, params, types, shift, zero_worm, repeat, !rank);
  for (int i=0; i<spin.bonds.size(); i++) {
    cout << "[" << spin.bond_type[i] << "," << spin.bonds[i][0] << "," << spin.bonds[i][1] << "]" << endl;
  }
}