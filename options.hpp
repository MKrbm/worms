/*
   worms: a simple worm code

   Copyright (C) 2013-2021 by Synge Todo <wistaria@phys.s.u-tokyo.ac.jp>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

// default & command line options

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <streambuf>
#include <sstream>

struct options {
  unsigned int L;
  unsigned int dim;
  double T;
  double H;
  double J1;
  double J2;
  double shift=0;
  unsigned int sweeps;
  unsigned int therm;
  std::string MN;
  std::vector<std::string> path_list;
  bool valid;
  unsigned int argc;


  static int my_strlen(char *input_string)
  {
      int i;
      for(i=0; input_string[i] != '\0'; i++);
      return i;
  }


  static std::vector<std::string> argv2vector(unsigned int argc, char *argv[]){
    std::vector<std::string> vec;

    for (int i=0; i<argc; i++){
      vec.push_back(argv[i]);
    }
    return vec;
  }

  options(unsigned int argc, char *argv[], unsigned int L_def, unsigned int dim_def, double T_def, std::string M_def)
  :options(argv2vector(argc, argv), L_def, dim_def, T_def, M_def){}

  options(std::vector<std::string> argv, unsigned int L_def, unsigned int dim_def, double T_def, std::string M_def)
  :L(L_def), T(T_def), H(0), sweeps(1 << 16), therm(sweeps >> 3), valid(true),J1(1), J2(1),
  dim(dim_def), MN(M_def), argc(argv.size())
  {
    auto argc = argv.size();
    for (int i=0; i<argc; ++i){
      auto str = argv[i];
      if (str.find("#") != std::string::npos){
        continue;
      }
      if (str.find("-L") != std::string::npos){
        if (++i<argc) L = std::atoi(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-D") != std::string::npos){
        if (++i<argc) dim = std::atoi(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-T") != std::string::npos){
        if (++i<argc) T = std::atof(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-H") != std::string::npos){
        if (++i<argc) H = std::atof(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-J1") != std::string::npos){
        if (++i<argc) J1 = std::atof(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-J2") != std::string::npos){
        if (++i<argc) J2 = std::atof(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-m") != std::string::npos){
        if (++i<argc) therm = std::atoi(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-n") != std::string::npos){
        if (++i<argc) sweeps = std::atoi(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-M") != std::string::npos){
        if (++i<argc) MN = argv[i];
        else usage();
        continue;
      }
      if (str.find("-PATH") != std::string::npos){
        if (++i<argc) path_list.push_back(argv[i]);
        else usage();
        continue;
      }
      if (str.find("-shift") != std::string::npos){
        if (++i<argc) shift = std::atof(argv[i].c_str());
        else usage();
        continue;
      }
      if (str.find("-h") != std::string::npos){
        usage(std::cout);
        continue;
      }
    }
  if (T <= 0 || sweeps == 0) {
    std::cerr << "invalid parameter\n"; usage(); return;
  }
  std::cout << "System Linear Size     = " << L << '\n'
            << "Temperature            = " << T << '\n'
            << "Magnetic Field         = " << H << '\n'
            << "MCS for Thermalization = " << therm << '\n'
            << "MCS for Measurement    = " << sweeps << '\n';
}


  void usage(std::ostream& os = std::cerr) {
    os << "[command line options]\n"
       << "  -L int    System Linear Size\n"
       << "  -T double Temperature\n"
       << "  -H double Magnetic Field\n"
       << "  -m int    MCS for Thermalization\n"
       << "  -n int    MCS for Measurement\n"
       << "  -h        this help\n";
    valid = false;
  }
};



struct readConfig : options{
  public:
  static std::vector<std::string> string2path(std::string path){
    std::ifstream t(path);
    std::string str_((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    std::cout << str_ << std::endl;
    std::vector<std::string> vec;
     
    std::stringstream check(str_);
     
    std::string intermediate;
    vec.push_back(path.c_str());
     
    while(getline(check, intermediate, ' '))
    {
      std::stringstream ss(intermediate);
      std::string tmp;
      while(getline(ss, tmp, '\n')){
        if (tmp != "")
        {
          vec.push_back(tmp.c_str());
        }
      }
    }
    return vec;
  }
  

  readConfig(std::string path ,unsigned int L_def, unsigned int dim_def, double T_def, std::string M_def)
  :options(string2path(path), L_def, dim_def, T_def, M_def)
  {}
};
