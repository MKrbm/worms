cmake_minimum_required(VERSION 3.0.0)

# needed for include gmp
# set(CMAKE_SYSTEM_PROCESSOR arm64) #only mac
# set(CMAKE_OSX_ARCHITECTURES arm64) # only mac



project(worms VERSION 0.1.0) # set project version
option(RANDOM_SEED "wether using random seed or not " ON)
option(NDEBUG "wether using random seed or not " ON)


#include find package
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# #find package for gmp
# find_package(GMP REQUIRED)
# find_package(GMPXX REQUIRED)

message("None = ${CMAKE_C_FLAGS}")
message("Debug = ${CMAKE_C_FLAGS_DEBUG}")
message("Release = ${CMAKE_C_FLAGS_RELEASE}")
message("GCC_DEBUG_FLAGS = ${GCC_DEBUG_FLAGS}")

# set(CMAKE_CXX_COMPILER "/usr/bin/g++")
# set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE PATH "" FORCE)



set(CMAKE_CXX_FLAGS_GPROF
    "-g -O3 -DNDEBUG -Wall -pg"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )

MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_GPROF)


add_subdirectory(./worms) #call /LoopAlgo/CMakeLists.txst



# main_app
add_executable(main_app main.cpp) # add executable
target_link_libraries(main_app worm) # link library


# debug_app
add_executable(debug debug.cpp) # add executable
target_link_libraries(debug worm) # link library
# target_compile_features(worm PRIVATE cxx_std_20)


configure_file(MainConfig.h.in ${PROJECT_SOURCE_DIR}/MainConfig.h)

target_include_directories(main_app PUBLIC
            "${PROJECT_BINARY_DIR}"
            )
target_include_directories(debug PUBLIC
            "${PROJECT_BINARY_DIR}"
            )

target_compile_features(debug PUBLIC cxx_std_14)
target_compile_features(main_app PUBLIC cxx_std_14)

# test
add_executable(test ./gmp/test.cpp)
target_compile_features(test PUBLIC cxx_std_14)
target_link_libraries(test worm) # link library
# target_link_libraries(test gmp gmpxx)