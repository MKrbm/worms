if(NOT DEVCORE_POSTFIX_CMAKE_INCLUDED)
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

  # Build type
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
    message("Inside if block")
  endif(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type : " ${CMAKE_BUILD_TYPE})

  # C++ standards
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  # enable_testing()
endif(NOT DEVCORE_POSTFIX_CMAKE_INCLUDED)
