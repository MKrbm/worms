cmake_minimum_required(VERSION 3.1)
project(bcl
    VERSION 1.0.0
    DESCRIPTION "balance condition library"
    LANGUAGES CXX)
# target_compile_features(bcl PRIVATE cxx_std_14)
add_library(bcl INTERFACE)
# include_directories(${PROJECT_SOURCE_DIR})
target_include_directories(bcl INTERFACE .)
set_target_properties(bcl
    PROPERTIES
    VERSION ${PROJECT_VERSION})