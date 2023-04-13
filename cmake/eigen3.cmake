find_package(Eigen3)
message("worms // searching eigen3")
if(Eigen3_FOUND)
  message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
  message(STATUS "Eigen3 libs: ${EIGEN3_LIBRARIES}")
  message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
else(Eigen3_FOUND)
  include(FetchContent)
  message(STATUS "not find eigen3  / fetch content")
  # fix eigen3
  FetchContent_Declare(
    eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
    )
    FetchContent_MakeAvailable(eigen3)
endif(Eigen3_FOUND)
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")

