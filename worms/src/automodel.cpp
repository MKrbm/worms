#include <lattice/graph_xml.hpp>
#include <lattice/basis.hpp>

#include "../include/automodel.hpp"






namespace model{

  base_lattice::base_lattice(int L, VVS bonds)
  :L(L), Nb(bonds.size()), N_op(1), bonds(bonds), bond_type(VS(bonds.size(), 0)), site_type(VS(L, 0)){}

  base_lattice::base_lattice(int L, VVS bonds, VS bond_type, VS site_type)
  :L(L), Nb(bonds.size()), N_op(num_type(bond_type)),bonds(bonds), bond_type(bond_type), site_type(site_type){
    VS tmp(bond_type);
    std::sort(tmp.begin(), tmp.end());
    if (tmp[0] != 0) {throw std::invalid_argument("bond type need to starts from 0");}
    if (tmp.back() != N_op - 1 + tmp[0]) {throw std::invalid_argument("bond types need to takes consective integer");}

    bond_t_size = VS(N_op, 0);
    type2bonds = vector<VVS>(N_op);
    for (int i=0; i<N_op; i++) for (int j=0; j<Nb; j++) if (bond_type[j]==i) {
      bond_t_size[i]++; 
      if (type2bonds[i].size() > 0 && type2bonds[i].back().size() != bonds[j].size()) throw std::invalid_argument("legsize should be consistent among the operator with the same type");
      type2bonds[i].push_back(bonds[j]);
      }
    }

  base_lattice::base_lattice(std::tuple<size_t, VVS, VS, VS> tp)
  :base_lattice(get<0>(tp), get<1>(tp), get<2>(tp), get<3>(tp)){}

  base_lattice::base_lattice(std::string basis_name, std::string cell_name, VS shapes, std::string file, bool print)
  :base_lattice(initilizer_xml(basis_name, cell_name, shapes, file, print))
  {}

    
  VVS generate_bonds(lattice::graph lattice){
      VVS bonds;
      for (int b=0; b<lattice.num_bonds(); b++) bonds.push_back({lattice.source(b), lattice.target(b)});
      for (int b=0; b<lattice.num_multis(); b++) bonds.push_back(lattice.multi(b));
      return bonds;
    }

  VS generate_bond_type(lattice::graph lattice){
    VS bond_type;
    for (int b=0; b<lattice.num_bonds(); b++) bond_type.push_back(lattice.bond_type(b));
    for (int b=0; b<lattice.num_multis(); b++) bond_type.push_back(lattice.multi_type(b));
    return bond_type;
  }

  VS generate_site_type(lattice::graph lattice){
    VS site_type;
    for (int b=0; b<lattice.num_sites(); b++){site_type.push_back(lattice.site_type(b));}
    return site_type;
  }

  size_t num_type(VS bond_type){
    std::sort(bond_type.begin(), bond_type.end());
    return std::distance(bond_type.begin(), std::unique(bond_type.begin(), bond_type.end()));
  }



  std::tuple<size_t, VVS, VS, VS> base_lattice::initilizer_xml(string basis_name, string cell_name, VS shapes, string file, bool print)
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
        if (shapes.size()!=1) {std::cerr << "Wrong number of shapes for 1D lattice"; exit(1);}
        lattice::graph lat(bs, cell, lattice::extent(shapes[0]));
        if (print) lat.print(std::cout);
        return make_tuple(lat.num_sites(), generate_bonds(lat), generate_bond_type(lat), generate_site_type(lat));
        break;
      }
    case 2:
      {
        if (shapes.size()!=2) {std::cerr << "Wrong number of shapes for 2D lattice"; exit(1);}
        lattice::graph lat(bs, cell, lattice::extent(shapes[0], shapes[1]));
        if (print) lat.print(std::cout);
        return make_tuple(lat.num_sites(), generate_bonds(lat), generate_bond_type(lat), generate_site_type(lat));
        break;
      }
      break;
    case 3:
      {
        if (shapes.size()!=3) {std::cerr << "Wrong number of shapes for 3D lattice"; exit(1);}
        lattice::graph lat(bs, cell, lattice::extent(shapes[0], shapes[1], shapes[2]));
        if (print) lat.print(std::cout);
        return make_tuple(lat.num_sites(), generate_bonds(lat), generate_bond_type(lat), generate_site_type(lat));
        break;
      }
    default:
      cerr << "Unsupported lattice dimension\n";
      exit(127);
      break;
    }
    return make_tuple(0, VVS(), VS(), VS());

    }
  /*
  definition of automodels.
  */

  //* default constructor
  template <class MC>
  base_model<MC>::base_model( model::base_lattice lat, 
              VS dofs, 
              std::string ham_path, 
              VD params, 
              VI types, 
              double shift, 
              bool zero_worm, 
              bool repeat,
              bool print)
  :base_lattice(lat)
  {
    //only accept single dof
    bool all_same = true;
    for (int i = 1; i < dofs.size(); ++i) if (dofs[i] != dofs[0]) {std::cerr << "dofs must be single valued\n"; exit(1);}

    //* prepare _sps_sites
    if (num_type(site_type) != dofs.size()) {std::cerr << "# of dofs doesn't match to # of site types\n"; exit(1);}
    for (int t : site_type) {_sps_sites.push_back(dofs[t]);}
    if (_sps_sites.size() != L) {std::cerr << "Something wrong with sps_sites\n"; exit(1);}


    // cout << "hi" << endl;
    //* raed all numpy files in given path.
    std::vector<std::string> path_list;
    get_npy_path(ham_path, path_list);
    std::sort(path_list.begin(), path_list.end());

    //* if repeat = true
    VI types_tmp;
    VD params_tmp;
    if (repeat){
      int r_cnt = N_op/types.size(); //repeat count
      if (r_cnt * types.size() != N_op) {std::cerr << "can not finish repeating types and params\n"; exit(1);}
      for (int i=0; i<r_cnt; i++) {
        types_tmp.insert(types_tmp.end(), types.begin(), types.end());
        params_tmp.insert(params_tmp.end(), params.begin(), params.end());
        for (auto &x : types) x += types.size();
      }
      if(print) cout << "repeat params " << r_cnt << " times." << endl;
      types = types_tmp;
      params = params_tmp;
    }



    //* check types
    if (params.size() != types.size()) {std::cerr << "size of params and types must match\n";exit(1);}
    VI _types(types);
    std::sort(_types.begin(), _types.end());
    int uniqueCount = std::unique(_types.begin(), _types.end()) - _types.begin();
    if ((size_t)N_op != 1+*std::max_element(_types.begin(), _types.end()) || N_op != uniqueCount)
    {  
      std::cerr << "types does not match requirements\n";
      exit(1);
    }

    //* load hamiltonians
    VVS dofs_list(N_op);
    for (int i=0; i<N_op; i++) {
      for (auto b : type2bonds[i][0]) {dofs_list[i].push_back(_sps_sites[site_type[b]]);} //size should be leg_size
      loperators.push_back(local_operator<MC>(type2bonds[i][0].size(), dofs_list[i][0]));  // local_operator only accepts one sps type yet. Also, currently only available for bond operator.
    }

    for (int p_i=0; p_i<path_list.size(); p_i++) {
      std::string path = path_list[p_i];
      auto pair = load_npy(path);
      VS shape = pair.first;
      VD data = pair.second;
      if (shape[0]!= shape[1]){ std::cerr << "require square matrix" << std::endl; exit(1); }
      size_t S = shape[0];
      auto& dof = dofs_list[types[p_i]];
      if (S != accumulate(dof.begin(), dof.end(), 1, multiplies<size_t>())) {
        std::cerr << "dimenstion of given matrix does not match to dofs ** legsize" << std::endl;
        std::cerr << "matrix size : " << S << std::endl; 
        exit(1); }

      if (print) std::cout << "hamiltonian is read from " << path << std::endl;
      local_operator<MCT>& loperator = loperators[types[p_i]];
      for (int i=0; i<shape[0]; i++) for (int j=0; j<shape[1]; j++)
      {
        auto x = data[i * shape[1] + j] * params[p_i];
        loperator._ham[i][j] += x;
      }
    }

    //* initial settings for local bond operators
    VD off_sets(N_op, shift);
    initial_setting(off_sets, 1E-8, zero_worm);

    //* calculate origin shift
    origin_shift = 0;
    for (int e=0; e< N_op; e++){
      origin_shift +=  shifts[e] *  bond_t_size[e];
    }
  }


  //* simple constructor
  template <class MC>
  base_model<MC>::base_model( model::base_lattice lat, 
              VS dofs, 
              std::vector<std::vector<std::vector<double>>> hams, 
              double shift, 
              bool zero_worm)
  :base_lattice(lat)
  {
    if (N_op != hams.size()) {std::cerr << "size of hams does not match to N_op\n"; exit(1);}
    for (int t : site_type) {_sps_sites.push_back(dofs[t]);}
    VVS dofs_list(N_op);
    for (int i=0; i<N_op; i++) {
      for (auto b : type2bonds[i][0]) {dofs_list[i].push_back(_sps_sites[site_type[b]]);} //size should be leg_size
      loperators.push_back(local_operator<MC>(type2bonds[i][0].size(), dofs_list[i][0])); // local_operator only accepts one 
      for (int j=0; j<hams[i].size(); j++) for (int k=0; k<hams[i][j].size(); k++) {
        loperators[i]._ham[j][k] = hams[i][j][k];
      }
    }

    //* initial settings for local bond operators
    VD off_sets(N_op, shift);
    initial_setting(off_sets, 1E-8, zero_worm);

    //* calculate origin shift
    origin_shift = 0;
    for (int e=0; e< N_op; e++){
      origin_shift +=  shifts[e] *  bond_t_size[e];
    }
  }

  template <class MC>
  void base_model<MC>::initial_setting(VD off_sets, double thres, bool zw){
    int i = 0;
    double tmp=0;
    for (local_operator<MCT> & h : loperators){
      h.set_ham(off_sets[i], thres, zw);
      shifts.push_back(h.ene_shift);
      i++;
    }
  }


template class model::base_model<bcl::heatbath>;
template class model::base_model<bcl::st2010>;
template class model::base_model<bcl::st2013>;
}

