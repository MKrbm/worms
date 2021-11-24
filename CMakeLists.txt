cmake_minimum_required(VERSION 3.0.0)


project(worms VERSION 0.1.0) # set project version
option(RANDOM_SEED "wether using random seed or not " ON)


message("None = ${CMAKE_C_FLAGS}")
message("Debug = ${CMAKE_C_FLAGS_DEBUG}")
message("Release = ${CMAKE_C_FLAGS_RELEASE}")

# set(Hello "-O3")

# set(CMAKE_BUILD_TYPE Hello) #Debug for -O0 flag

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



