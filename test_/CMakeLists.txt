add_executable(test ./test.cpp)
target_link_libraries(test worm) # link library

add_executable(npy ./npy_test.cpp)
target_link_libraries(npy worm) # link library

add_executable(header ./header_test.cpp ./explict.cpp ./header.h)
target_include_directories(header PUBLIC "${PROJECT_BINARY_DIR}" ${FetchContent_includes})



add_executable(lattice_test ./lattice_test.cpp )
target_link_libraries(lattice_test worm)
target_include_directories(lattice_test PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(remove_template ./remove_template.cpp )
target_link_libraries(remove_template worm)
target_include_directories(remove_template PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(3site ./3site.cpp )
target_link_libraries(3site worm)
target_include_directories(3site PUBLIC "${PROJECT_BINARY_DIR}")

add_executable(example1 ./example1.cpp )
target_link_libraries(example1 lattice)
# target_include_directories(example1 PUBLIC "${PROJECT_BINARY_DIR}" ${FetchContent_includes})