#include "../include/load_npy.hpp"
#include <npy.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <dirent.h>

std::pair<std::vector<unsigned long>, std::vector<double>> load_npy(std::string path) {

  try
  {
    std::vector<unsigned long> shape;
    std::vector<double> data;
    bool fortran_order;
    shape.clear();
    data.clear();
    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    return std::make_pair(shape, data);
  }
  catch(...)
  {
    std::cerr << "I/O error while reading npy file : " << path << "\n";
    exit(127);
  }

}

void get_npy_path(std::string dir_path, std::vector<std::string>& npy_path){
  DIR *di;
  char *ptr1,*ptr2;
  int retn;
  struct dirent *dir;
  di = opendir(dir_path.c_str()); 
  if (di)
  {
      while ((dir = readdir(di)) != NULL)
      {
          ptr1=strtok(dir->d_name,"."); ptr2=strtok(NULL,".");
          if(ptr2!=NULL) {
            retn=strcmp(ptr2,"npy"); 
            if(retn==0) {std::string path(ptr1); path += ".npy"; npy_path.push_back(dir_path +  "/" + path);}
            }
      }
      closedir(di);
  }else{
    std::cout << "cannot open folder : " << dir_path << std::endl;
  }
}