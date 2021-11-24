#include "../include/loop.hpp"



solver::solver(double beta, heisenberg1D model) 
#if DEBUG==1
  :L(model.L), beta(beta), uftree(0), state(L, 1), model(model), rand_src(2021),
  dist(0,L-1),binary_dice(0,1), bonds(model.bonds), front_group(L, -1), end_group(L, -1)
#else
  :L(model.L), beta(beta), uftree(0), state(L, 1), model(model), 
  rand_src(static_cast <unsigned> (time(0))),
  dist(0,L-1),binary_dice(0,1), bonds(model.bonds), front_group(L, -1), end_group(L, -1)
#endif
{
  #if DEBUG==1
  std::cout << "sovler is initialized" << std::endl;
  // srand (static_cast <unsigned> (time(0)));
  #else
  srand(static_cast <unsigned> (time(0)));
  #endif

}

void solver::step(){

  double tau = 0;
  double tau_prime = 0;
  double r;
  int r_bond; // randomly choosen bond
  int ktau_label = 0; // label of tau for the last kink at each loop - 1. Next kink label starts from ktau_label

  /* initialization */
  uftree.initialize(); //initilize uftree (par, size, etc ..)
  initialize_cuts();
  /* end of initilization */

  std::fill(end_group.begin(), end_group.end(), -1);
  std::fill(front_group.begin(), front_group.end(), -1);

  while (true){
    r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

    tau_prime = tau - log(r)/model.rho;

    // assigin kinks between [tau, tau_prime] to cuts.
    check_kink_and_flip(ktau_label, tau_prime);
    
    if (tau_prime > beta) break;

    // generate random integer within [0, .., L-1]
    r_bond = dist(rand_src);

    // If spins at choosen bond is antiparallel, assigin cut at the bond.
    if (check_ap(r_bond)){
      cuts.push_back(r_bond);
      cuts_tau.push_back(tau_prime);
      kink_in_cuts.push_back(1);

      //union the groups which are connected with vertical line (see some figure in the text book)
      assiginAndUnion(bonds[r_bond][0], bonds[r_bond][1]);
    }
    tau = tau_prime;
    // std::cout << "random site : " << r_bond << std::endl;
  }
  // end of while loop

  //we need to unite the forefron and backmost nodes after tau reaching beta.
  for (int i=0; i < model.Nb; i++){
    uftree.unite(front_group[i], end_group[i]);
  }

  initialize_kinks(); //resize kinks to zero
  uftree.initialize_flips(); //initialize all elements in flips vector to 0


  //update spin and kinks
  update();
}

void solver::update(){

  int n_op = cuts.size();
  int par;

  // finish
  for (int i=0; i < n_op; i++){
    if (findNflip(2*i) * findNflip(2*i+1) * kink_in_cuts[i] == -1){
      kinks.push_back(cuts[i]);
      kinks_tau.push_back(cuts_tau[i]);
    }
  }
  kink_in_cuts.resize(0);

  // update spin at \tau = 0
  int node;
  for (int i=0; i<L; i++){
    node = end_group[i];
    state[i] *= findNflip(node);
  }
}

int solver::findNflip(int x){
  if (x==-1) return binary_dice(rand_src) * 2 - 1;
  x = uftree.find(x);
  if (uftree.flips[x] == 0){
    uftree.flips[x] = binary_dice(rand_src)*2 - 1;
  }
  return uftree.flips[x];
}


void solver::read_kinks_csv(std::string filename){
  /* 
  filename is filename of .csv file whose path is ../LoopAlgorithm/src/csv/filename
  */

  std::ifstream mycsv;
  std::string exePath = getExePath();
  std::string path = exePath + "/../LoopAlgorithm/src/csv/" + filename;
  mycsv.open(path);
  std::cout << "start reading from : "<< path << std::endl;
  std::cout << "\nkinks are initialized\n" <<std::endl;

  /* initialize to 0 */
  kinks.resize(0);
  kinks_tau.resize(0);

  if(!mycsv.is_open()){
    std::cout << "The current directory is : " <<  exePath << std::endl;   
    return;
  }


  std::string line;
  std::string tmp;

  double tau;
  int label;
  while (getline(mycsv, line))
  {
    std::stringstream ss;
    ss << line;

    std::getline(ss, tmp, ',');
    tau = std::stof(tmp);
    if (tau > beta) throw std::runtime_error("tau must be smaller than beta\n");
    kinks_tau.push_back(tau);

    std::getline(ss, tmp, ',');
    label = std::stoi(tmp);
    if (label > model.Nb) throw std::runtime_error("given bond label is invalid \n");
    kinks.push_back(label);

  }
  mycsv.close();
}


int solver::check_kink_and_flip(int& tau_label, double tau_prime){
  /* 
  return tau_label where one start to look for the next time.
  */


  //If kinks isn't defined yet, return 0
  if (!kinks.size()) return 0;

  double ktau = kinks_tau[tau_label];
  int bond_label = kinks[tau_label];
  int kink_size = kinks.size();
  int i = 0, N = 0;

  int s1, s2;
  while((ktau < tau_prime)&&(tau_label <= kink_size-1)){

    s1 = bonds[bond_label][0];
    s2 = bonds[bond_label][1];

    state[s1] *= -1;
    state[s2] *= -1;


    cuts.push_back(bond_label);
    cuts_tau.push_back(ktau);

    assiginAndUnion(s1, s2); // assgin new nodes and unite if necessary.

    kink_in_cuts.push_back(-1);

    tau_label++;
    i++;
    ktau = kinks_tau[tau_label];
    bond_label = kinks[tau_label];
    /* flip state according to off-diagonal operator */
  }
  return i;
}

void solver::assiginAndUnion(int site1, int site2){

  int N = uftree.add_group(2); //this return current number of nodes in tree
  if (end_group[site1]==-1){
    end_group[site1] = N-2; //label is current # of nodes - 2.
  }else{
    uftree.unite(N-2, front_group[site1]); //If there is already a operator at the site, unite new nodes to the operator.
  }
  if (end_group[site2]==-1){
    end_group[site2] = N-2; //label is current # of nodes - 2.
  }else{
    uftree.unite(N-2, front_group[site2]);
  }
  front_group[site1] = N-1; 
  front_group[site2] = N-1; 

}

void LoopUnionFindTree::findNflip(int x, int& y, int& flip){
  check_index(x);
  if (flips[x] != 0){
    flip = flips[x];
    return;
  }else{
    if (par[x] == x){
      y = x;
      flip = dist(rand_src)*2-1;
      flips[x] = flip;
      return;
    }else{
      findNflip(par[x], y, flip);
      par[x] = y;
      flips[x] = flip;
      return;
    }
  }

}
