cmake_minimum_required(VERSION 3.6)
# Note : must be included by master CMakeLists.txt
# Tests in static libraries might not get registered, see https://github.com/onqtam/doctest/blob/master/doc/markdown/faq.md#why-are-my-tests-in-a-static-library-not-getting-registered
# For this reason, and because it is interesting to have individual
# test executables for each library, it is suggested not to put tests directly in the libraries (even though doctest advocates this usage)
# Creating multiple executables is of course not mandatory, and one could use the same executable with various command lines to filter what tests to run.

add_executable(failtest failtest.cpp)
target_link_libraries(failtest doctest::doctest)

add_test(
    # Use some per-module/project prefix so that it is easier to run only tests for this module
    NAME BP.failtest
    COMMAND failtest ${TEST_RUNNER_PARAMS}
)
set_tests_properties(
    BP.failtest
    PROPERTIES
        WILL_FAIL TRUE # We expect this test to fail
)

add_executable(successtest successtest.cpp)
target_link_libraries(successtest doctest::doctest ${PROJECT_NAME}::foo)

add_test(
    NAME BP.successtest
    COMMAND successtest ${TEST_RUNNER_PARAMS}
)

# SVM Performance Comparison Test
add_executable(svm_performance_test
    models/svm-performance-test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../source/math/matrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../source/models/support-vector-machine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../source/models/support-vector-machine-basic.cpp
)

target_include_directories(svm_performance_test
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include
        ${CMAKE_CURRENT_LIST_DIR}/../source
)

target_link_libraries(svm_performance_test
    PRIVATE
        doctest::doctest
        fmt::fmt
)

add_test(
    NAME ML.svm_performance_test
    COMMAND svm_performance_test ${TEST_RUNNER_PARAMS}
)
