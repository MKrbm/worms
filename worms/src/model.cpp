#include "../include/model.hpp"

// using namespace bond;

namespace model{
  std::vector<BOND> generate_bonds(lattice::graph lattice){
      std::vector<BOND> bonds;
      for (int b=0; b<lattice.num_bonds(); b++){
        std::vector<size_t> tmp(2);
        tmp[0] = lattice.source(b);
        tmp[1] = lattice.target(b);
        bonds.push_back(tmp);
      }
      return bonds;
    }

  std::vector<size_t> generate_bond_type(lattice::graph lattice){
    std::vector<size_t> bond_type;
    for (int b=0; b<lattice.num_bonds(); b++) bond_type.push_back(lattice.bond_type(b));
    return bond_type;
  }

  size_t num_type(std::vector<size_t> bond_type){
    std::sort(bond_type.begin(), bond_type.end());
    auto it = std::unique(bond_type.begin(), bond_type.end());
    return std::distance(bond_type.begin(), it);
  }
}
