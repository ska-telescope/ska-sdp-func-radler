# SPDX-License-Identifier: LGPL-3.0-only

include(unittest)

add_unittest(
  radler_math runtests.cc test_dijkstra_splitter.cc test_peak_finder.cc
  test_rms_image.cc test_component_optimization.cc)
