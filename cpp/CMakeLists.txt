# Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

# CMake function to keep directory structure when installing headers.
function(install_headers_with_directory HEADER_LIST)
  foreach(HEADER ${HEADER_LIST})
    string(REGEX MATCH ".*\/" DIR ${HEADER})
    install(FILES ${HEADER}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/include/radler/${DIR})
  endforeach(HEADER)
endfunction(install_headers_with_directory)

add_library(${PROJECT_NAME} SHARED "")

set(RADLER_FILES
    radler.cc
    deconvolution_table.cc
    component_list.cc
    image_set.cc
    algorithms/deconvolution_algorithm.cc
    algorithms/generic_clean.cc
    algorithms/iuwt_deconvolution_algorithm.cc
    # algorithms/ls_deconvolution.cc // TODO: Complete or remove this class.
    algorithms/more_sane.cc
    algorithms/multiscale_algorithm.cc
    algorithms/parallel_deconvolution.cc
    algorithms/python_deconvolution.cc
    algorithms/simple_clean.cc
    algorithms/subminor_loop.cc
    algorithms/threaded_deconvolution_tools.cc
    algorithms/iuwt/image_analysis.cc
    algorithms/iuwt/iuwt_decomposition.cc
    algorithms/iuwt/iuwt_mask.cc
    algorithms/multiscale/multiscale_transforms.cc
    math/peak_finder.cc
    math/rms_image.cc
    utils/casa_mask_reader.cc)

# A number of files perform the 'core' high-performance floating point
# operations. In these files, NaNs are avoided and thus -ffast-math is allowed.
# Note that visibilities can be NaN hence this can not be turned on for all
# files.
set_source_files_properties(
  image_set.cpp
  algorithms/generic_clean.cpp
  algorithms/multiscale_algorithm.cpp
  algorithms/threaded_deconvolution_tools.cpp
  algorithms/simple_clean.cpp
  algorithms/subminor_loop.cpp
  algorithms/multiscale/multiscale_transforms.cpp
  PROPERTIES COMPILE_FLAGS -ffast-math)

# Using pybind11 requires using -fvisibility=hidden. See
# https://pybind11.readthedocs.io/en/stable/faq.html
set_source_files_properties(algorithms/python_deconvolution.cpp
                            PROPERTIES COMPILE_FLAGS -fvisibility=hidden)

target_sources(${PROJECT_NAME} PRIVATE ${RADLER_FILES})
target_link_libraries(${PROJECT_NAME} ${RADLER_TARGET_LIBS})
target_include_directories(${PROJECT_NAME} SYSTEM
                           PUBLIC ${RADLER_TARGET_INCLUDE_DIRS})

# Allows including the paths relative to base algorithm, in line with Google
# style
# https://google.github.io/styleguide/cppguide.html#Names_and_Order_of_Includes
target_include_directories(
  ${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_compile_options(radler PRIVATE -O3 -Wall -Wzero-as-null-pointer-constant)

if(NOT PORTABLE)
  if(DEFINED TARGET_CPU)
    target_compile_options(radler BEFORE PRIVATE -march=${TARGET_CPU})
  else()
    check_cxx_compiler_flag("-march=native" COMPILER_HAS_MARCH_NATIVE)
    if(COMPILER_HAS_MARCH_NATIVE)
      target_compile_options(radler BEFORE PRIVATE -march=native)
    else()
      message(
        WARNING "The compiler doesn't support -march=native for your CPU.")
    endif()
  endif()
endif()

if(NOT COMPILE_AS_EXTERNAL_PROJECT)
  include(GNUInstallDirs)
  install(TARGETS ${PROJECT_NAME}
          LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR})
else()
  install(TARGETS ${PROJECT_NAME}
          LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

set(PUBLIC_HEADERS
    component_list.h
    radler.h
    deconvolution_settings.h
    deconvolution_table.h
    deconvolution_table_entry.h
    image_set.h
    algorithms/multiscale/multiscale_transforms.h)

install_headers_with_directory("${PUBLIC_HEADERS}")

if(NOT COMPILE_AS_EXTERNAL_PROJECT)
  configure_file(${PROJECT_SOURCE_DIR}/cmake/config/radler-config.cmake.in
                 ${PROJECT_BINARY_DIR}/CMakeFiles/radler-config.cmake @ONLY)
  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/config/radler-config-version.cmake.in
    ${PROJECT_BINARY_DIR}/CMakeFiles/radler-config-version.cmake @ONLY)

  # Install configuration files
  install(FILES ${PROJECT_BINARY_DIR}/CMakeFiles/radler-config.cmake
                ${PROJECT_BINARY_DIR}/CMakeFiles/radler-config-version.cmake
          DESTINATION ${CMAKE_INSTALL_PREFIX}/share/radler)
endif()