cmake_minimum_required(VERSION 3.0.0)

# include(cmake/prefix.cmake)
project(worms VERSION 1.0.0 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# C++ standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# c and c++ compiler
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc CACHE PATH "")
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++ CACHE PATH "")


unset(CMAKE_Fortran_COMPILER CACHE)

message(STATUS "${CMAKE_Fortran_COMPILER} is available")
message(STATUS "current directory is ${PROJECT_BINARY_DIR}")

include(CheckLanguage)
check_language(Fortran)
message(STATUS "${CMAKE_Fortran_COMPILER} is available")
option(RANDOM_SEED "wether using random seed or not " ON)
include_directories(${worms_SOURCE_DIR})

#include libraries
# include(bcl)
# include(eigen3)
# include(npy)

message(STATUS "after include directory: ${CMAKE_Fortran_COMPILER} is available")
message("None = ${CMAKE_C_FLAGS}")
message("Debug = ${CMAKE_C_FLAGS_DEBUG}")
message("Release = ${CMAKE_C_FLAGS_RELEASE}")
message("GCC_DEBUG_FLAGS = ${GCC_DEBUG_FLAGS}")

set(CMAKE_CXX_FLAGS_GPROF
    "-g -O3 -DNDEBUG -Wall -pg"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )

MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_GPROF)
add_subdirectory(./worms) #call /LoopAlgo/CMakeLists.txst


# add executable

# search for MPI
find_package(MPI REQUIRED)
FIND_PACKAGE(Boost COMPONENTS mpi serialization REQUIRED)

FOREACH(name main debug main_MPI)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} worm  ${Boost_LIBRARIES} MPI::MPI_CXX)
    target_include_directories(${name} PUBLIC
            "${PROJECT_BINARY_DIR}"
            "${Boost_INCLUDE_DIRS}"
        )
ENDFOREACH(name)


add_subdirectory(test)

configure_file(MainConfig.h.in MainConfig.h)