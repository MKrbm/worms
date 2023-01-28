set (test_src
     example
     worm
    )

FOREACH(name ${test_src})
  add_executable(${name}-gtest ${name}.cpp)
  target_link_libraries(${name}-gtest worm gtest_main)
  target_include_directories(${name}-gtest PUBLIC "${PROJECT_BINARY_DIR}")
  list(APPEND gtest_targets ${name}-gtest)
  add_test(NAME ${name} COMMAND  ${gtest_targets})
ENDFOREACH(name)

enable_testing()
