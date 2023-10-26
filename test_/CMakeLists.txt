cmake_minimum_required(VERSION 3.10)

project(HelloMPI)

# Ensure C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_COMPILER "/usr/local/bin/mpicxx")
project(HelloMPI)

# Find MPI
find_package(MPI REQUIRED COMPONENTS C CXX)


# Check the found paths
message(STATUS "MPI_CXX_FOUND = ${MPI_CXX_FOUND}")
message(STATUS "MPI_CXX_INCLUDE_DIRS = ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")

message(STATUS "MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
message(STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")

# Include directories
# include_directories(${MPI_INCLUDE_PATH})

# Compile the hello_world program
add_executable(hello hello_world.cpp)
target_link_libraries(hello MPI::MPI_CXX)

# If using dynamic MPI libraries, we should set these flags
# if(MPI_COMPILE_FLAGS)
#   set_target_properties(hello PROPERTIES
#     COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
# endif()
#
# if(MPI_LINK_FLAGS)
#   set_target_properties(hello PROPERTIES
#     LINK_FLAGS "${MPI_LINK_FLAGS}")
# endif()
