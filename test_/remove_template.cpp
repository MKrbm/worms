// #include <lattice/graph_xml.hpp>
// #include <lattice/basis.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <libconfig.h++>
#include <dirent.h>
#include <filesystem>
#include <unistd.h>
#include <automodel.hpp>
#include <exec2.hpp>
// namespace boost { namespace property_tree {
//     class ptree;
//   }
// }
using namespace std;
using namespace libconfig;

int main() {


  // cout << spin.bonds << endl;

  char tmp[256];
  auto _ = getcwd(tmp, 256);
  cout << tmp << endl;

  Config cfg;
  cfg.setAutoConvert(true);

  try { cfg.readFile("/home/project/config/model.cfg");}
  catch(const FileIOException &fioex)
  {
    cerr << "I/O error while reading file." << endl;
    return(EXIT_FAILURE);
  }

  const Setting& root = cfg.getRoot();
  string model_name = root["model"];
  cout << "model name is \t : \t" << model_name << endl;
  const Setting& model_config = root["models"][model_name];
  const Setting& shape_cfg = model_config.lookup("length");
  const Setting& dofs_cfg = model_config.lookup("dofs");
  const Setting& params_cfg = model_config.lookup("params");
  const Setting& types_cfg = model_config.lookup("types");

  int dof;
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


  model_config.lookupValue("file", file); 
  model_config.lookupValue("basis", basis); 
  model_config.lookupValue("cell", cell); 
  model_config.lookupValue("ham_path", ham_path); 
  model_config.lookupValue("repeat", repeat);
  model_config.lookupValue("shift", shift);
  model_config.lookupValue("zero_worm", zero_worm);

  model::base_lattice lat(basis, cell, shapes, file, true);
  model::base_model<> spin(lat, dofs, ham_path, params, types, shift, zero_worm, repeat);

  // cout << spin_state::StateFunc::pows_array(4, 4) << endl;
  // cout << spin.bonds << endl;
  // cout << spin.bond_type << endl;
  // const Setting& settings = root["mc_settings"];
  // exe_worm(spin, settings);
}