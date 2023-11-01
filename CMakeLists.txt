cmake_minimum_required(VERSION 3.14)

# include(cmake/prefix.cmake)
project(worms VERSION 1.0.0 LANGUAGES CXX)

# cmake will find modules from this path with include command
# you don't need to run include if there is no option to download from github
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(MPI REQUIRED COMPONENTS CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# c and c++ compiler
# set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc CACHE PATH "")
# set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++ CACHE PATH "")

# debug when use find_package
set(CMAKE_FIND_DEBUG_MODE OFF) #turn on to print out debug info for find_package

# cmake use rpath for linking shared libraries at runtime
set(CMAKE_SKIP_RPATH OFF)
set(BUILD_RPATH_USE_ORIGIN ON)

# set working_dir (main_MPI will execute from this directory)
set(WORKING_DIR "${CMAKE_BINARY_DIR}" CACHE  STRING "The working directory for the project")

# cmake use system path for finding libraries
# if these variables set to ON (True) and if you installed boost package via miniconda, find_package will look for miniconda path first and result in error
# set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)


message(STATUS "CMAKE_FIND_USE_CMAKE_SYSTEM_PATH = ${CMAKE_FIND_USE_CMAKE_SYSTEM_PATH}")


unset(CMAKE_Fortran_COMPILER CACHE)

include(CheckLanguage)
check_language(Fortran)
option(RANDOM_SEED "wether using random seed or not " ON)

include(CTest)
enable_testing()

message("None = ${CMAKE_C_FLAGS}")
message("Debug = ${CMAKE_C_FLAGS_DEBUG}")
message("Release = ${CMAKE_C_FLAGS_RELEASE}")
message("GCC_DEBUG_FLAGS = ${GCC_DEBUG_FLAGS}")

set(CMAKE_CXX_FLAGS_GPROF
    "-g -O3 -DNDEBUG -Wall -pg"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )

MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_GPROF)

# search for  
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

message(STATUS "MPI_CXX_FOUND = ${MPI_CXX_FOUND}")
message(STATUS "MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")


add_executable(hello test_/hello_world.cpp)
target_link_libraries(hello MPI::MPI_CXX)

include_directories(${worms_SOURCE_DIR})

add_subdirectory(./worms) #call /LoopAlgo/CMakeLists.txst
add_subdirectory(./gtest)

FIND_PACKAGE(Boost COMPONENTS serialization filesystem REQUIRED)

FOREACH(name main_MPI debug)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} worm  ${Boost_LIBRARIES})
    target_include_directories(${name} PUBLIC
            "${PROJECT_BINARY_DIR}"
            "${Boost_INCLUDE_DIRS}"
        )
ENDFOREACH(name)

# message(STATUS "MPI_CXX_FOUND = ${MPI_CXX_FOUND}")
# message(STATUS "MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}")
# message(STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")

# target_link_libraries(worm-gtest worm)
# target_include_directories(worm-gtest PUBLIC "${PROJECT_BINARY_DIR}")
# add_test(NAME test COMMAND  worm-gtest)

# add_subdirectory(examples)
configure_file(MainConfig.h.in MainConfig.h)
