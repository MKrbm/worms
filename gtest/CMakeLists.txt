set (test_src
     example
     worm
    )
# enable_testing()
include(GoogleTest)
FOREACH(name ${test_src})
  add_executable(${name}-gtest ${name}.cpp)
  target_link_libraries(${name}-gtest worm gtest_main)
  target_include_directories(${name}-gtest PUBLIC "${PROJECT_BINARY_DIR}")
  gtest_discover_tests(${name}-gtest)
  # list(APPEND gtest_targets ${name}-gtest)
  # add_test(
  #   NAME ${name} 
  #   COMMAND ${name}-gtest
  #   CONFIGURATIONS Release
  #   WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
ENDFOREACH(name)

