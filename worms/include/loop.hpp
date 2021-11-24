#ifndef __loop__
#define __loop__

#pragma once
#include <string.h>
#include <iostream>
#include <uftree.hpp>
#include <vector>
#include <random>
#include <fstream>
#include <ostream>
#include <strstream>
#include <sstream>

#ifdef __APPLE__
#  include <mach-o/dyld.h>
#endif
#if defined(_WIN32)
#  include <windows.h>
#else
  #include <unistd.h>
#endif

#include <mach-o/dyld.h>

#include <filesystem>
#include <unistd.h>

#include <ctime>
#include <math.h> 

#include "model.hpp"

/* inherit UnionFindTree and add find_and_flip function*/
class LoopUnionFindTree: public UnionFindTree
{
	std::mt19937 rand_src;
  std::uniform_int_distribution<> dist;
public : 
	std::vector<int> flips;
  LoopUnionFindTree(int n) 
    :UnionFindTree(n), rand_src(2021), dist(0,1)
    {}
	
	void findNflip(int x, int& y, int& flip);
	void initialize() override {
    N = 0;
    par.resize(0);
    sizes.resize(0);
		flips.resize(0);
  }
	void initialize_flips(){
		flips.resize(get_N());
		std::fill(flips.begin(), flips.end(), 0);
	}
};


class solver{
private:
  int L; 

  /*

  two nodes in unionfind are assigned every time new cut is assiged. 
  node label = index in cuts / 2
  node whose label%2 = 0 is assigend to end_groups and 1 is to front_groups;

  */

  std::vector<int> front_group; //vector of length L. Each element of index i corresponds to the foremost bond(group) of site i. -1 means no off-diagonal operator has been assigend yet.
  std::vector<int> end_group;  // the backmost bond.
  
  std::vector<int> cuts;  // vector holding the bond label of graph 3 (off-diagonal operator) in order of occurrence.
  std::vector<double> cuts_tau; //corresponding time series.

  std::vector<int> kinks;  // vector holding the bond label of kinks in order of occurrence.
  std::vector<double> kinks_tau;  //corresponding time series.

  std::vector<int> kink_in_cuts; // holds where are the kinks in the cuts vector. kink = -1, cuts = 1;

  std::vector<int> flip_groups; // assigin 1 or -1 independently for each group. 
  /* for random seed
  std::random_device seed_gen;
  std::default_random_engine engine;
  */
   
  std::mt19937 rand_src;
  std::uniform_int_distribution<> dist;
  std::uniform_int_distribution<> binary_dice;

	void initialize_cuts(){
		cuts.resize(0);
		cuts_tau.resize(0);
	}

	void initialize_kinks(){
		kinks.resize(0);
		kinks_tau.resize(0);
	}

public:
  solver(double beta, heisenberg1D model);
  heisenberg1D model;
  double beta; //revese temperature
  // LoopUnionFindTree& uftree_p; //class for union find.
	LoopUnionFindTree uftree;

  int check_kink_and_flip(int& tau_label, double tau_prime);
  void read_kinks_csv(std::string filename); 
  void step(); //fucntion for one montecarlo step
  void update();
  int findNflip(int x);

	int get_kink_num(){
		return kinks.size();
	}

  int check_ap(int bond_label){
		std::vector<int> &bond = bonds[bond_label];
		// int j = bonds[bond_label][1];
		#if DEBUG==1
		std::cout << "anitiparallel ? : " << (state[bond[0]] == -state[bond[1]])<< std::endl;
		#endif

		return state[bond[0]] == -state[bond[1]];
  }

private:
  std::vector<int> state;
  std::vector<std::vector<int>> bonds;

  void assiginAndUnion(int site1, int site2);


};


// class LoopUnionFindTree: public UnionFindTree
// {
// 	std::mt19937 rand_src;
//   std::uniform_int_distribution<> dist;
// public : 
// 	std::vector<int> flips;
//   LoopUnionFindTree(int n) 
//     :UnionFindTree(n), rand_src(2021), dist(0,1)
//     {}
	
// 	void findNflip(int x, int& y, int& flip);
// };

inline std::string getExePath(){
  char buffer[1024];
  uint32_t size = sizeof(buffer);
  std::string path;

#if defined (WIN32)
  GetModuleFileName(nullptr,buffer,size);
  path = buffer;
#elif defined (__APPLE__)
  namespace fs = std::__fs::filesystem;
  if(_NSGetExecutablePath(buffer, &size) == 0)
  {
  path = buffer;
  }
#elif defined(UNIX) || defined(unix) || defined(__unix) || defined(__unix__)
  throw std::runtime_error("UNIX is not available \n");
  exit;
  int byte_count = readlink("/proc/self/exe", buffer, size);
  if (byte_count  != -1)
  {
  path = std::string(buffer, byte_count);
  }
#endif

  fs::path p = path;
  return static_cast<std::string>(p.parent_path());
}

#endif 