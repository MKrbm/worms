#include <lattice/graph_xml.hpp>
#include <lattice/basis.hpp>

#include "../include/automodel.hpp"

namespace model{
  template <size_t MAX_L, class MC>
  model::base_model::base_model(std::string file, std::string basis_name, std::string cell_name)
  {
    std::ifstream is(file);
    boost::property_tree::ptree pt;
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
}
