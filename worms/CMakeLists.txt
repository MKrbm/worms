project(worm_lib
    VERSION 1.0
    DESCRIPTION "worm_algorithm library"
    LANGUAGES CXX)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(ALPSCore REQUIRED alea)

include(libconfig)
include(eigen3)
include(npy)
include(googletest)


add_library(worm STATIC
    ./src/operator.cpp
    ./src/load_npy.cpp
    ./src/automodel.cpp
    ./src/state_func.cpp
    ./src/worm.cpp
    ./src/autoobservable.cpp
    ./src/funcs.cpp
    ./src/exec_parallel.cpp
    )

add_subdirectory(./lib/bcl) #call /lib/bcl/CMakeLists.txst
add_subdirectory(./lib/lattice) #call /lib/bcl/CMakeLists.txst


# Get the directory of the first library in ALPSCore_LIBRARIES
# get_filename_component(ALPSCore_LIBRARY_DIR "${ALPSCore_LIBRARIES}" DIRECTORY)

# # Make an educated guess about the include directory
# set(ALPSCore_GUESSED_INCLUDE_DIR "${ALPSCore_LIBRARY_DIR}/../include/ALPSCore/")

# message(STATUS "Guessed ALPSCore include dir: ${ALPSCore_LIBRARY_DIR}")

target_link_libraries( worm
    MPI::MPI_CXX  
    bcl 
    config++ 
    lattice 
    ${ALPSCore_LIBRARIES} 
    gtest_main
) # link library



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
