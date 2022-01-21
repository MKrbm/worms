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
  unsigned int sweeps;
  unsigned int therm;
  std::string MN;
  bool valid;
  unsigned int argc;


  static int my_strlen(char *input_string)
  {
      int i;
      for(i=0; input_string[i] != '\0'; i++);
      return i;
  }


  static std::vector<char*> argv2vector(unsigned int argc, char *argv[]){
    std::vector<char*> vec;

    for (int i=0; i<argc; i++){
      vec.push_back(argv[i]);
    }
    return vec;
  }

  options(unsigned int argc, char *argv[], unsigned int L_def, unsigned int dim_def, double T_def, std::string M_def)
  :options(argv2vector(argc, argv), L_def, dim_def, T_def, M_def){}

  options(std::vector<char*> argv, unsigned int L_def, unsigned int dim_def, double T_def, std::string M_def)
  :L(L_def), T(T_def), H(0), sweeps(1 << 16), therm(sweeps >> 3), valid(true),J1(1), J2(1),
  dim(dim_def), MN(M_def), argc(argv.size())
  {
  for (unsigned int i = 1; i < argc; ++i) {
    switch (argv[i][0]) {
    case '-' :
      switch (argv[i][1]) {
      case 'L' :
        if (++i == argc) { usage(); return; }
        L = std::atoi(argv[i]); break;
      case 'D' :
        if (++i == argc) { usage(); return; }
        dim = std::atoi(argv[i]); break;
      case 'T' :
        if (++i == argc) { usage(); return; }
        T = std::atof(argv[i]); break;
      case 'H' :
        if (++i == argc) { usage(); return; }
        H = std::atof(argv[i]); break;
      case 'm' :
        if (++i == argc) { usage(); return; }
        therm = std::atoi(argv[i]); break;
      case 'n' :
        if (++i == argc) { usage(); return; }
        sweeps = std::atoi(argv[i]); break;
      case 'M' :
        if (++i == argc) { usage(); return; }
        MN = argv[i]; break;
      case 'J' :
        switch (argv[i][2]){
          case '1':
            if (++i == argc) { usage(); return; }
            J1 = std::atof(argv[i]); break;
          case '2':
            if (++i == argc) { usage(); return; }
            J2 = std::atof(argv[i]); break;
          default :
            usage(); return;
        }
        break;
      case 'h' :
        usage(std::cout); return;
      default :
        usage(); return;
      }
      break;
    default :
      usage(); return;
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
  static std::vector<char*> string2path(std::string path){
    std::ifstream t(path);
    std::string str_((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    std::cout << str_ << std::endl;
    std::vector<char*> vec;
     
    std::stringstream check(str_);
     
    std::string intermediate;
    char *c = new char[sizeof(const_cast<char*>(path.c_str()))];
    std::strcpy(c, const_cast<char*>(path.c_str()));
    vec.push_back(c);
     
    // Tokenizing w.r.t. space ' '
    while(getline(check, intermediate, ' '))
    {
      std::stringstream ss(intermediate);
      std::string tmp;
      while(getline(ss, tmp, '\n')){
        if (tmp != "")
        {
          char *c = new char[sizeof(const_cast<char*>(tmp.c_str()))];
          std::strcpy(c, const_cast<char*>(tmp.c_str()));
          vec.push_back(c);
        }
      }
    }
    return vec;
  }

  readConfig(std::string path ,unsigned int L_def, unsigned int dim_def, double T_def, std::string M_def)
  :options(string2path(path), L_def, dim_def, T_def, M_def)
  {}
};
