find_package(GTest QUIET)
message(STATUS "GTest_FOUND: ${GTest_FOUND}")
if(GTest_FOUND)
  message(STATUS "Found GTest: ${GTest_INCLUDE_DIRS}")
  return()
endif()
message(STATUS "GTest not found, downloading...")
option(BUILD_GMOCK OFF)
option(INSTALL_GTEST OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
