cmake_minimum_required(VERSION 3.0.0)

# needed for include gmp
# set(CMAKE_SYSTEM_PROCESSOR arm64) #only mac
# set(CMAKE_OSX_ARCHITECTURES arm64) # only mac



# include(cmake/prefix.cmake)
project(worms VERSION 1.0.0 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(RANDOM_SEED "wether using random seed or not " ON)
option(NDEBUG "wether using random seed or not " ON)


#include find package
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# #find package for gmp
# find_package(GMP REQUIRED)
# find_package(GMPXX REQUIRED)

#include findpackages
include_directories(${worms_SOURCE_DIR})

#include libraries
include(bcl)
include(lattice)
include(eigen3)

FetchContent_MakeAvailable(bcl)
include_directories(${FetchContent_includes})


message("None = ${CMAKE_C_FLAGS}")
message("Debug = ${CMAKE_C_FLAGS_DEBUG}")
message("Release = ${CMAKE_C_FLAGS_RELEASE}")
message("GCC_DEBUG_FLAGS = ${GCC_DEBUG_FLAGS}")

# set(CMAKE_CXX_COMPILER "/usr/bin/g++")
# set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE PATH "" FORCE)



set(CMAKE_CXX_FLAGS_GPROF
    "-g -O3 -DNDEBUG -Wall -pg"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )

MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_GPROF)
add_subdirectory(./worms) #call /LoopAlgo/CMakeLists.txst


# add executable
FOREACH(name main debug)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} worm)
    target_include_directories(${name} PUBLIC
            "${PROJECT_BINARY_DIR}"
        )
    target_compile_features(${name} PUBLIC cxx_std_14)
ENDFOREACH(name)


configure_file(MainConfig.h.in MainConfig.h)


# test
add_executable(test ./gmp/test.cpp)
target_compile_features(test PUBLIC cxx_std_14)
target_link_libraries(test worm) # link library



