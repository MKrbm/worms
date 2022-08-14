// #include <boost/property_tree/ptree.hpp>
#include <lattice/graph_xml.hpp>
#include <lattice/basis.hpp>
// #include <boost/regex.hpp>
// #include "header.h"
#include <iostream>
#include <string>
#include <fstream>

// namespace boost { namespace property_tree {
//     class ptree;
//   }
// }

class FOO{
  public:
};

int main(int argc, char **argv) {
  // std::cout << "Hi" << std::endl;
  std::string file = "../test_/lattices.xml";
  std::string basis_name = "chain lattice";
  std::string cell_name = "nnn1d";
  std::size_t length = 4;
  if (argc > 1) {
    if (argc == 4 || argc == 5) {
      file = argv[1];
      basis_name = argv[2];
      cell_name = argv[3];
      if (argc == 5) length = atoi(argv[4]);
    } else {
      std::cerr << "Error: " << argv[0] << " xmlfile basis cell [length]\n";
      exit(127);
    }
  }


  // std::cout << "Hi" << std::endl;
  std::ifstream is(file);
  boost::property_tree::ptree pt;
  // FOO a;
  read_xml(is, pt);
  lattice::basis bs;
  read_xml(pt, basis_name, bs);
  lattice::unitcell cell;
  read_xml(pt, cell_name, cell);
  switch (cell.dimension()) {
  case 1:
    { lattice::graph lat(bs, cell, lattice::extent(length)); lat.print(std::cout); }
    break;
  case 2:
    { lattice::graph lat(bs, cell, lattice::extent(length, 2)); lat.print(std::cout); }
    break;
  case 3:
    { lattice::graph lat(bs, cell, lattice::extent(length, length, length)); lat.print(std::cout); }
    break;
  default:
    std::cerr << "Unsupported lattice dimension\n";
    exit(127);
    break;
  }

  
}