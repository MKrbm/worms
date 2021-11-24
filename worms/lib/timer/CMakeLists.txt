cmake_minimum_required(VERSION 3.1)
project(uftree_lib
    VERSION 1.0.0
    DESCRIPTION "timer library"
    LANGUAGES CXX)
add_library(timer INTERFACE)
# target_compile_features(timer PRIVATE cxx_std_20)
# target_compile_options(timer PRIVATE -O3 -Wall)
target_include_directories(timer INTERFACE ./include)