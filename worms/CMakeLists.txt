cmake_minimum_required(VERSION 3.1)
project(worm_lib
    VERSION 1.0
    DESCRIPTION "worm_algorithm library"
    LANGUAGES CXX)

option(BUILD_SHARED_LIBS  "Enable shared library" OFF)
include(libconfig)
FetchContent_MakeAvailable(libconfig)

add_library(worm STATIC
    ./src/state.cpp
    ./src/operator.cpp
    ./src/model.cpp
    ./src/load_npy.cpp
    )

add_subdirectory(./lib/bcl) #call /lib/bcl/CMakeLists.txst
target_link_libraries(worm bcl config++) # link library


message(STATUS "include from path : ${FetchContent_includes}")
target_include_directories(worm PUBLIC ./include ${FetchContent_includes})
# target_precompile_headers(worm PUBLIC
#     ./include/pch.hpp
#   )
# target_include_directories(worm INTERFACE ./include)
# target_compile_options(worm PRIVATE -O3 -Wall)

set_target_properties(worm
    PROPERTIES
    VERSION ${PROJECT_VERSION})