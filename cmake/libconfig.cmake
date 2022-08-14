message(STATUS "inside")
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
endif(libconfig_FOUND)