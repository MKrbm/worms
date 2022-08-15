// #include <lattice/graph_xml.hpp>
// #include <lattice/basis.hpp>
#include <automodel.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <libconfig.h++>
#include <dirent.h>
#include <filesystem>
#include <unistd.h>

// namespace boost { namespace property_tree {
//     class ptree;
//   }
// }
using namespace std;
using namespace libconfig;

template<class T> ostream& operator<<(ostream& os, const vector<T>& vec) {
    os << "[ ";
    for ( const T& item : vec )
        os << item << ", ";
    os << "]"; return os;
}

int main() {

  // cout << spin.bonds << endl;

  char tmp[256];
  getcwd(tmp, 256);
  cout << tmp << endl;

  Config cfg;
  try
  {
    cfg.readFile("/home/user/project/test_/model.cfg");
  }
  catch(const FileIOException &fioex)
  {
    cerr << "I/O error while reading file." << endl;
    return(EXIT_FAILURE);
  }
  const Setting& root = cfg.getRoot();
  const Setting &model_config = root["models"]["majumdar_ghosh"];
  const Setting &shape_cfg = model_config.lookup("length");
  const Setting &params_cfg = model_config.lookup("params");
  const Setting &types_cfg = model_config.lookup("types");

  int dof;
  string file, basis, cell, ham_path;
  bool repeat_params; // true if repeat params.
  vector<size_t> shapes;
  vector<int> types, params;
  for (int i=0; i<shape_cfg.getLength(); i++) {int tmp = shape_cfg[i]; shapes.push_back(tmp);}
  for (int i=0; i<params_cfg.getLength(); i++) {params.push_back(params_cfg[i]);}
  for (int i=0; i<types_cfg.getLength(); i++) {types.push_back(types_cfg[i]);}


  model_config.lookupValue("file", file); 
  model_config.lookupValue("basis", basis); 
  model_config.lookupValue("cell", cell); 
  model_config.lookupValue("ham_path", ham_path); 
  model_config.lookupValue("dof", dof); 
  model_config.lookupValue("repeat_params", repeat_params);




  model::base_model<> spin(basis, cell, shapes, file, dof, ham_path, params, types ,false);





  // cout << "Hi" << endl;
  // string file = "../test_/lattices.xml";
  // string basis_name = "chain lattice";
  // string cell_name = "nnn1d";
  // size_t length = 4;
  // if (argc > 1) {
  //   if (argc == 4 || argc == 5) {
  //     file = argv[1];
  //     basis_name = argv[2];
  //     cell_name = argv[3];
  //     if (argc == 5) length = atoi(argv[4]);
  //   } else {
  //     cerr << "Error: " << argv[0] << " xmlfile basis cell [length]\n";
  //     exit(127);
  //   }
  // }


  // // cout << "Hi" << endl;
  // ifstream is(file);
  // boost::property_tree::ptree pt;
  // // FOO a;
  // read_xml(is, pt);
  // lattice::basis bs;
  // read_xml(pt, basis_name, bs);
  // lattice::unitcell cell;
  // read_xml(pt, cell_name, cell);
  // switch (cell.dimension()) {
  // case 1:
  //   { lattice::graph lat(bs, cell, lattice::extent(length)); lat.print(cout); }
  //   break;
  // case 2:
  //   { lattice::graph lat(bs, cell, lattice::extent(length, 2)); lat.print(cout); }
  //   break;
  // case 3:
  //   { lattice::graph lat(bs, cell, lattice::extent(length, length, length)); lat.print(cout); }
  //   break;
  // default:
  //   cerr << "Unsupported lattice dimension\n";
  //   exit(127);
  //   break;
  // }

  
}