# SPDX-License-Identifier: LGPL-3.0-only

# Generic function for adding a radler unittest. It creates a 'test' that builds
# the unittest and a test that runs it. Arguments:
#
# * First argument: Module name, e.g., 'math'.
# * Next arguments: Source file names.
#
# Return value:
#
# * Sets TEST_NAME to the unit test name in the parent scope.

function(add_unittest MODULE_NAME)
  set(TEST_NAME "unittests_${MODULE_NAME}")
  set(TEST_NAME
      ${TEST_NAME}
      PARENT_SCOPE)
  set(FILENAMES ${ARGN})

  # Add boost dynamic link flag for all test files.
  # https://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/boost_test/usage_variants.html
  # Without this flag, linking is incorrect and boost performs duplicate
  # delete() calls after running all tests, in the cleanup phase.
  set_source_files_properties(${FILENAMES} PROPERTIES COMPILE_DEFINITIONS
                                                      "BOOST_TEST_DYN_LINK")

  add_executable(${TEST_NAME} ${FILENAMES})
  target_link_libraries(${TEST_NAME} radler
                        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  target_include_directories(
    ${TEST_NAME} PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>)

  # Add test for automatically (re)building the test if needed. The
  # RESOURCE_LOCK prevents concurrent build executions.
  add_test(build_${TEST_NAME} ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
           --target ${TEST_NAME})
  set_tests_properties(
    build_${TEST_NAME} PROPERTIES FIXTURES_SETUP ${TEST_NAME} RESOURCE_LOCK
                                  build)

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} -f JUNIT -k ${TEST_NAME}.xml
                                     --catch_system_error=yes)
  set_tests_properties(${TEST_NAME} PROPERTIES LABELS unit FIXTURES_REQUIRED
                                               ${TEST_NAME})

endfunction()
