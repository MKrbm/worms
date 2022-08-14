cmake_minimum_required(VERSION 3.1)
project(worm_lib
    VERSION 1.0
    DESCRIPTION "worm_algorithm library"
    # URL関係のライブラリをインストールしないと動かないことがあるので
    # コメントアウトしておきます(2019/12/02 修正)
    # HOMEPAGE_URL "https://example.com"
    LANGUAGES CXX)
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS  "Enable shared library" OFF)
include(libconfig)
FetchContent_MakeAvailable(libconfig)
# set(BUILD_SHARED_LIBS OFF)

add_library(worm STATIC
    ./src/state.cpp
    ./src/operator.cpp
    ./src/model.cpp
    )

add_subdirectory(./lib/bcl) #call /lib/bcl/CMakeLists.txst
target_link_libraries(worm bcl config++) # link library


# target_compile_features(worm PRIVATE cxx_std_14)
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