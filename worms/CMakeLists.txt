cmake_minimum_required(VERSION 3.1)
project(worm_lib
    VERSION 1.0
    DESCRIPTION "worm_algorithm library"
    # URL関係のライブラリをインストールしないと動かないことがあるので
    # コメントアウトしておきます(2019/12/02 修正)
    # HOMEPAGE_URL "https://example.com"
    LANGUAGES CXX)
add_library(worm STATIC
    ./src/worm.cpp 
    ./src/model.cpp
    ./src/BC.cpp
    ./src/state.cpp
    ./src/heisenberg1D.cpp
    )

add_subdirectory(./lib/uftree) #call /uftree/CMakeLists.txst


target_link_libraries(worm uftree) # link library



# target_compile_features(worm PRIVATE cxx_std_14)
target_include_directories(worm INTERFACE ./include)
# target_compile_options(worm PRIVATE -O3 -Wall)

set_target_properties(worm
    PROPERTIES
    VERSION ${PROJECT_VERSION})