# - Find GMPXX
# Find the GMPXX includes and libraries
# This module defines
#  GMPXX_INCLUDE_DIR, where to find gmpxx.h
#  GMPXX_LIBRARY, libgmpxx.so or libgmpxx.a
#  GMPXX_FOUND, If false, do not try to use GMPXX.
#
# Availbale targets:
#  GMPXX::GMPXX

find_path(GMPXX_INCLUDE_DIR gmpxx.h
  PATHS
    ENV GMPXX_ROOT
    ENV GMPXX_INCLUDE_DIR
    ${GMPXX_ROOT}
    /usr
    /usr/local
    $ENV{HOME}/.local
  PATH_SUFFIXES
    include
  )

find_library(GMPXX_LIBRARY
  NAMES
    gmpxx
  PATHS
    ENV GMPXX_ROOT
    ENV GMPXX_LIB_DIR
    ${GMPXX_ROOT}
    /usr
    /usr/local
  PATH_SUFFIXES
    lib
  )

mark_as_advanced(GMPXX_INCLUDE_DIR GMPXX_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMPXX
  REQUIRED_VARS
    GMPXX_INCLUDE_DIR
    GMPXX_LIBRARY
  )

if (NOT TARGET GMP::GMP)
  find_package(GMP REQUIRED)
endif()

if(GMPXX_FOUND AND NOT TARGET GMPXX::GMPXX)
  add_library(GMPXX::GMPXX UNKNOWN IMPORTED)
  set_target_properties(GMPXX::GMPXX PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${GMPXX_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GMPXX_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES GMP::GMP
    )
endif()

