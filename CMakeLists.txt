cmake_minimum_required(VERSION 3.0.0)

# include(cmake/prefix.cmake)
project(worms VERSION 1.0.0 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# C++ standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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
include(eigen3)
message(STATUS "after include: ${CMAKE_Fortran_COMPILER} is available")

include(lattice)
include(npy)
FetchContent_MakeAvailable(lattice)
FetchContent_MakeAvailable(npy)

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
FOREACH(name main debug heisenberg SS ladder MG)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} worm)
    target_include_directories(${name} PUBLIC
            "${PROJECT_BINARY_DIR}"
        )
ENDFOREACH(name)

add_subdirectory(test_)

configure_file(MainConfig.h.in MainConfig.h)