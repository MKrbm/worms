message(STATUS "inside")
option(BUILD_SHARED_LIBS  "Enable shared library" OFF)
find_package(libconfig)
if(libconfig_FOUND)
  message(STATUS "libconfig was found path = ${libconfig_CONFIG}")
else(libconfig_FOUND)
  include(FetchContent)
  message(STATUS "not found libconfig  / fetch content")
  FetchContent_Declare(
    libconfig
    GIT_REPOSITORY https://github.com/hyperrealm/libconfig.git
    )
  FetchContent_MakeAvailable(libconfig)
endif(libconfig_FOUND)
