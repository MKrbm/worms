include(FetchContent)
FetchContent_Declare(
  npy
  GIT_REPOSITORY https://github.com/llohse/libnpy.git
)
list(APPEND FetchContent_includes "${PROJECT_BINARY_DIR}/_deps/npy-src/include")
list(APPEND FetchContents npy)