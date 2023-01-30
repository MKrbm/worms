#!/bin/bash

# bash script for installation of libconfig

working_dir=$(eval echo ~user)
cd $working_dir
git clone https://github.com/MKrbm/libconfig.git --branch v1.7.3
cd libconfig/
mkdir build && cd build
cmake .. -G Ninja -DBUILD_SHARED_LIBS=OFF
ninja && ninja install
cd ../../ && rm -rf libconfig/