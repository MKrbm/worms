set (test_src
     wobs
     lattice
     model
     example
     lops
    )


# enable_testing()

include_directories(${worms_SOURCE_DIR})

include(GoogleTest)
FOREACH(name ${test_src})
  add_executable(${name}-gtest ${name}.cpp)
  target_link_libraries(${name}-gtest worm gtest_main)
  target_include_directories(${name}-gtest PUBLIC "${PROJECT_BINARY_DIR}")
  gtest_discover_tests(${name}-gtest 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
ENDFOREACH(name)


set(worm
    worm
    worm_wobs
    )

FOREACH(name ${worm})
  add_executable(${name}-gtest ${name}.cpp)
  target_link_libraries(${name}-gtest worm gtest_main)
  target_include_directories(${name}-gtest PUBLIC "${PROJECT_BINARY_DIR}")
ENDFOREACH(name)
  

