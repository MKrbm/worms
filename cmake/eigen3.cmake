find_package(Eigen3)
message("worms // searching eigen3")
message("Eigen3 config path  : ${Eigen3_CONFIG}")
if(Eigen3_FOUND)
  message(STATUS "eigen3 was found path = ${Eigen3_SOURCE_DIR}")
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

