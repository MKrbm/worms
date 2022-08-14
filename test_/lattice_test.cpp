// #include <lattice/graph_xml.hpp>
// #include <lattice/basis.hpp>
// #include <automodel.hpp>
#include <iostream>
// #include <string>
// #include <fstream>
// #include <libconfig.h++>

// namespace boost { namespace property_tree {
//     class ptree;
//   }
// }
using namespace std;
// using namespace libconfig;

// template<class T> ostream& operator<<(ostream& os, const vector<T>& vec) {
//     os << "[ ";
//     for ( const T& item : vec )
//         os << item << ", ";
//     os << "]"; return os;
// }

int main() {

  // model::base_model<> spin(10, std::vector<model::BOND>(5, std::vector<size_t>(2, 2)));
  // cout << spin.bonds << endl;

  cout << "Hello" << endl;
  // Config cfg;
  // try
  // {
  //   cfg.readFile("/home/user/project/test_/model.cfg");
  // }
  // catch(const FileIOException &fioex)
  // {
  //   std::cerr << "I/O error while reading file." << std::endl;
  //   return(EXIT_FAILURE);
  // }
  // const Setting& root = cfg.getRoot();
  // const Setting &model_config = root["model"];
  // const Setting &size_config = model_config["length"];
  // std::string file, basis, cell;
  // size_t l1, l2, l3;
  // model_config.lookupValue("file", file); 
  // model_config.lookupValue("basis", basis); 
  // model_config.lookupValue("cell", cell); 


  // size_config.lookupValue("l1", l1); 
  // size_config.lookupValue("l2", l2);
  // size_config.lookupValue("l3", l3); 

  


  // std::cout << "Hi" << std::endl;
  // std::string file = "../test_/lattices.xml";
  // std::string basis_name = "chain lattice";
  // std::string cell_name = "nnn1d";
  // std::size_t length = 4;
  // if (argc > 1) {
  //   if (argc == 4 || argc == 5) {
  //     file = argv[1];
  //     basis_name = argv[2];
  //     cell_name = argv[3];
  //     if (argc == 5) length = atoi(argv[4]);
  //   } else {
  //     std::cerr << "Error: " << argv[0] << " xmlfile basis cell [length]\n";
  //     exit(127);
  //   }
  // }


  // // std::cout << "Hi" << std::endl;
  // std::ifstream is(file);
  // boost::property_tree::ptree pt;
  // // FOO a;
  // read_xml(is, pt);
  // lattice::basis bs;
  // read_xml(pt, basis_name, bs);
  // lattice::unitcell cell;
  // read_xml(pt, cell_name, cell);
  // switch (cell.dimension()) {
  // case 1:
  //   { lattice::graph lat(bs, cell, lattice::extent(length)); lat.print(std::cout); }
  //   break;
  // case 2:
  //   { lattice::graph lat(bs, cell, lattice::extent(length, 2)); lat.print(std::cout); }
  //   break;
  // case 3:
  //   { lattice::graph lat(bs, cell, lattice::extent(length, length, length)); lat.print(std::cout); }
  //   break;
  // default:
  //   std::cerr << "Unsupported lattice dimension\n";
  //   exit(127);
  //   break;
  // }

  
}