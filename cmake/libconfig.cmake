# Initial attempt to find the package on the system
find_package(libconfig QUIET)

# Define the path for dependencies
set(DEPS_DIR "${CMAKE_BINARY_DIR}/_deps")

if(NOT libconfig_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    libconfig
    GIT_REPOSITORY https://github.com/hyperrealm/libconfig.git
    GIT_TAG        a06736788dc7607711a252e67af665ceb0dad656  # Specify the tag or commit as needed
    SOURCE_DIR     "${DEPS_DIR}/libconfig-src" # Source directory
    BINARY_DIR     "${DEPS_DIR}/libconfig-build" # Binary directory
  )
	FetchContent_MakeAvailable(libconfig)
else()
  message(STATUS "libconfig found: ${libconfig_CONFIG}")
endif()
