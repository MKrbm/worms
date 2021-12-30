find_package(Eigen3)
<<<<<<< HEAD
if(Eigen3_FOUND)
message("found")
message("EIGEN3_USE_FILE  =  ${EIGEN3_USE_FILE}")
else(Eigen3_FOUND)
message("not found")
=======
message("EIGEN3_USE_FILE = ${EIGEN3_USE_FILE}")
message("CMAKE_CURRENT_LIST_DIR = ${CMAKE_CURRENT_LIST_DIR}")

>>>>>>> 71444c3ef8d291752919e16b10ae69df7272d5da
  include(FetchContent)
  FetchContent_Declare(
    eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
    SOURCE_SUBDIR none
  )
  list(APPEND FetchContent_includes "${PROJECT_BINARY_DIR}/_deps/eigen3-src")
  list(APPEND FetchContents eigen3)
if(Eigen3_FOUND)
  include(${EIGEN3_USE_FILE})
else(Eigen3_FOUND)
endif(Eigen3_FOUND)
