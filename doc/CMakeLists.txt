find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

set(RADLER_SO_PATH ${CMAKE_BINARY_DIR}/python)

set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_GENERATED} @ONLY)

# This will be the main output of our command
set(DOXYGEN_INDEX_FILE "${DOXYGEN_OUTPUT_DIR}/xml/index.xml")

add_custom_command(
  OUTPUT ${DOXYGEN_INDEX_FILE}
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_GENERATED}
  WORKING_DIRECTORY ${DOXYGEN_OUTPUT_DIR}
  MAIN_DEPENDENCY Doxyfile
  COMMENT "Generating doxygen documentation")

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/html)

add_custom_target(
  doc
  ${CMAKE_COMMAND}
  -E
  env
  RADLER_SO_PATH=${RADLER_SO_PATH}
  ${SPHINX_EXECUTABLE}
  -b
  html
  ${SPHINX_SOURCE}
  ${SPHINX_BUILD}
  # Tell Breathe where to find the Doxygen output
  -Dbreathe_projects.Radler=${DOXYGEN_OUTPUT_DIR}/xml
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating documentation with Sphinx"
  DEPENDS ${DOXYGEN_INDEX_FILE} pyradler)