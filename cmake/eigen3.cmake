find_package(Eigen3)
if(Eigen3_FOUND)
else(Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
    SOURCE_SUBDIR none
  )

  list(APPEND FetchContent_includes "${CMAKE_BINARY_DIR}/_deps/eigen3-src")
  list(APPEND FetchContents eigen3)
  message(STATUS "not find eigen3  / fetch content")
endif(Eigen3_FOUND)
