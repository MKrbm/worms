#include <lattice/graph_xml.hpp>
#include <lattice/basis.hpp>

#include "../include/automodel.hpp"

namespace model{
  
VVS generate_bonds(lattice::graph lattice){
    VVS bonds;
    for (int b=0; b<lattice.num_bonds(); b++) bonds.push_back({lattice.source(b), lattice.target(b)});
    return bonds;
  }

std::vector<size_t> generate_bond_type(lattice::graph lattice){
  std::vector<size_t> bond_type;
  for (int b=0; b<lattice.num_bonds(); b++) bond_type.push_back(lattice.bond_type(b));
  return bond_type;
}

size_t num_type(std::vector<size_t> bond_type){
  std::sort(bond_type.begin(), bond_type.end());
  return std::distance(bond_type.begin(), std::unique(bond_type.begin(), bond_type.end()));
}

std::tuple<size_t, VVS, VS> base_lattice::initilizer_xml(string basis_name, string cell_name, VS shapes, string file, bool print)
{
  ifstream is(file);
  boost::property_tree::ptree pt;
  read_xml(is, pt);
  lattice::basis bs;
  read_xml(pt, basis_name, bs);
  lattice::unitcell cell;
  read_xml(pt, cell_name, cell);
  switch (cell.dimension()) {
  case 1:
    {
      lattice::graph lat(bs, cell, lattice::extent(shapes[0]));
      if (print) lat.print(std::cout);
      return make_tuple(lat.num_sites(), generate_bonds(lat), generate_bond_type(lat));
      break;
    }
  case 2:
    {
      lattice::graph lat(bs, cell, lattice::extent(shapes[0], shapes[1]));
      if (print) lat.print(std::cout);
      return make_tuple(lat.num_sites(), generate_bonds(lat), generate_bond_type(lat));
      break;
    }
    break;
  case 3:
    {
      lattice::graph lat(bs, cell, lattice::extent(shapes[0], shapes[1], shapes[2]));
      if (print) lat.print(std::cout);
      return make_tuple(lat.num_sites(), generate_bonds(lat), generate_bond_type(lat));
      break;
    }
  default:
    cerr << "Unsupported lattice dimension\n";
    exit(127);
    break;
  }
  return make_tuple(0, VVS(), VS());
}

base_lattice::base_lattice(string basis_name, string cell_name, VS shapes, string file, bool print)
:base_lattice(initilizer_xml(basis_name, cell_name, shapes, file, print))
{}
template class base_model<bcl::heatbath>;
}

