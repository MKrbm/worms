include(FetchContent)
FetchContent_Declare(
  npy
  GIT_REPOSITORY https://github.com/MKrbm/libnpy.git
)
list(APPEND FetchContent_includes "${CMAKE_BINARY_DIR}/_deps/npy-src/include")
list(APPEND FetchContents npy)