find_package(Eigen3)
if(Eigen3_FOUND)
message("found")
message("EIGEN3_USE_FILE  =  ${EIGEN3_USE_FILE}")
else(Eigen3_FOUND)
message("not found")
  include(FetchContent)
  FetchContent_Declare(
    eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
    SOURCE_SUBDIR none
  )
  list(APPEND FetchContent_includes "${PROJECT_BINARY_DIR}/_deps/eigen3-src")
  list(APPEND FetchContents eigen3)
endif(Eigen3_FOUND)
