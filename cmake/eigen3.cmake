find_package(Eigen3)
message("worms // searching eigen3")
if(Eigen3_FOUND)
  message(STATUS "eigen3 was found path = ${eigen3_SOURCE_DIR}")
else(Eigen3_FOUND)
  include(FetchContent)
  message(STATUS "not find eigen3  / fetch content")
  # fix eigen3
  FetchContent_Declare(
    eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
  )
  set(EIGEN_BUILD_DOC OFF)
  set(BUILD_TESTING OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  set( OFF)
  # list(APPEND FetchContent_includes "${CMAKE_BINARY_DIR}/_deps/eigen3-src")
  list(APPEND FetchContents Eigen3::Eigen)
  FetchContent_MakeAvailable(eigen3)
endif(Eigen3_FOUND)

