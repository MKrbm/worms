cmake_minimum_required(VERSION 3.1)
project(worm_lib
    VERSION 1.0
    DESCRIPTION "worm_algorithm library"
    LANGUAGES CXX)

include(libconfig)
include(eigen3)
include(npy)


# link alpscore
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
find_package(ALPSCore REQUIRED alea)

#* list all targets
# function(getAllSubdirs dir dirs)
#     message("called with dir (${dir}) and dirs (${dirs})")
#     set(_dirs "")
#     # get subdirectories for dir
#     get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
#     # iterate any found subdirectories
#     foreach(subdir ${subdirs})
#         # append each sub directory
#         list(APPEND _dirs ${subdir})
#         set(${dirs} _dirs PARENT_SCOPE)
#         getAllSubdirs(${subdir} ${_dirs})
#     endforeach()
# endfunction()
# set(dirs ".")
# getAllSubDirs(. ${dirs})
# message("all dirs " ${dirs})


# if(NOT (TARGET Eigen3::Eigen))
#   message(SEND_ERROR "Attempt to link to non-existent library Eigen3::Eigen")
# endif()

add_library(worm STATIC
    ./src/state.cpp
    ./src/operator.cpp
    ./src/load_npy.cpp
    ./src/automodel.cpp
    ./src/state_func.cpp
    ./src/worm.cpp
    )

add_subdirectory(./lib/bcl) #call /lib/bcl/CMakeLists.txst
add_subdirectory(./lib/lattice) #call /lib/bcl/CMakeLists.txst



target_link_libraries(worm bcl config++ lattice ${ALPSCore_LIBRARIES}) # link library


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