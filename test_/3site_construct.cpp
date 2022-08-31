#include <iostream>
#include <string>
#include <fstream>
#include <libconfig.h++>
#include <dirent.h>
#include <filesystem>
#include <unistd.h>
#include <automodel.hpp>
#include <exec2.hpp>
#include <options.hpp>
#include <argparse.hpp>
#include <funcs.hpp>
using namespace std;
using namespace libconfig;

using namespace std;


int main(int argc, char **argv) {
  char tmp[256];
  auto _ = getcwd(tmp, 256);
  cout << tmp << endl;

  Config cfg;
  cfg.setAutoConvert(true);

  try { cfg.readFile("/home/user/project/config/model.cfg");}
  catch(const FileIOException &fioex)
  {
    cerr << "I/O error while reading file." << endl;
    return(EXIT_FAILURE);
  }

  const Setting& root = cfg.getRoot();
  string model_name = root["model"];
  cout << "model name is \t : \t" << model_name << endl;
  const Setting& mcfg = root["models"][model_name];
  const Setting& shape_cfg = mcfg.lookup("length");
  const Setting& params_cfg = mcfg.lookup("params");
  const Setting& types_cfg = mcfg.lookup("types");
  const Setting& dofs_cfg = mcfg.lookup("dofs");


  double shift;
  string file, basis, cell, ham_path;
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
  repeat = (bool) mcfg.lookup("repeat");
  shift = (double) mcfg.lookup("shift");
  zero_worm = (bool) mcfg.lookup("zero_worm");

  cout << file << endl;

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

  //* argparse  
  argparse::ArgumentParser parser("test", "argparse test program", "Apache License 2.0");

  parser.addArgument({"-L1"}, "set shape[0]");
  parser.addArgument({"-L2"}, "set shape[1]");
  parser.addArgument({"-L3"}, "set shape[2]");
  parser.addArgument({"-T"}, "set temperature");

  auto args = parser.parseArgs(argc, argv);

  shapes[0] = args.safeGet<size_t>("L1", shapes[0]);
  shapes[1] = args.safeGet<size_t>("L2", shapes[1]);
  shapes[2] = args.safeGet<size_t>("L3", shapes[2]);
  T = args.safeGet<double>("T", T);

  cout << "zero_wom : " << (zero_worm ? "YES" : "NO") << endl;
  cout << "repeat : " << (repeat ? "YES" : "NO") << endl;


  //* finish argparse

  model::base_lattice lat(basis, cell, shapes, file, true);
  model::base_model<bcl::st2013> spin(lat, dofs, ham_path, params, types, shift, zero_worm, repeat);
  cout << lat.bonds << endl;
  exe_worm(spin, T, sweeps, therms, cutoff_l, fix_wdensity);  
  
}