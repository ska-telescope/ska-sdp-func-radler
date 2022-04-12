// SPDX-License-Identifier: LGPL-3.0-only

#include "deconvolution_table.h"

#include <array>

#include <boost/test/unit_test.hpp>

#include "test/smartptr.h"

namespace radler {

BOOST_AUTO_TEST_SUITE(deconvolutiontable)

BOOST_AUTO_TEST_CASE(constructor) {
  const size_t kTableSize = 42;

  DeconvolutionTable table(kTableSize, kTableSize);

  BOOST_TEST(table.OriginalGroups().size() == kTableSize);
  for (const DeconvolutionTable::Group& group : table.OriginalGroups()) {
    BOOST_TEST(group.empty());
  }

  BOOST_TEST_REQUIRE(table.DeconvolutionGroups().size() == kTableSize);
  for (int index = 0; index < int(kTableSize); ++index) {
    BOOST_TEST(table.DeconvolutionGroups()[index] ==
               std::vector<int>(1, index));
  }

  BOOST_TEST((table.begin() == table.end()));
  BOOST_TEST(table.Size() == 0);
}

BOOST_AUTO_TEST_CASE(single_deconvolution_group) {
  DeconvolutionTable table(7, 1);
  const std::vector<std::vector<int>> kExpectedGroups{{0, 1, 2, 3, 4, 5, 6}};
  BOOST_TEST_REQUIRE(table.DeconvolutionGroups() == kExpectedGroups);
}

BOOST_AUTO_TEST_CASE(multiple_deconvolution_groups) {
  DeconvolutionTable table(7, 3);
  const std::vector<std::vector<int>> kExpectedGroups{
      {0, 1, 2}, {3, 4}, {5, 6}};
  BOOST_TEST_REQUIRE(table.DeconvolutionGroups() == kExpectedGroups);
}

BOOST_AUTO_TEST_CASE(too_many_deconvolution_groups) {
  DeconvolutionTable table(7, 42);
  const std::vector<std::vector<int>> kExpectedGroups{{0}, {1}, {2}, {3},
                                                      {4}, {5}, {6}};
  BOOST_TEST_REQUIRE(table.DeconvolutionGroups() == kExpectedGroups);
}

BOOST_AUTO_TEST_CASE(add_entries) {
  DeconvolutionTable table(3, 1);

  std::array<test::UniquePtr<DeconvolutionTableEntry>, 3> entries;
  entries[0]->original_channel_index = 1;
  entries[1]->original_channel_index = 0;
  entries[2]->original_channel_index = 1;

  for (test::UniquePtr<DeconvolutionTableEntry>& entry : entries) {
    table.AddEntry(entry.take());
    // table.AddEntry(std::move(entry));
  }

  // Check if the OriginalGroups have the correct size and correct entries.
  const std::vector<DeconvolutionTable::Group>& original_groups =
      table.OriginalGroups();
  BOOST_TEST_REQUIRE(original_groups.size() == 3);

  BOOST_TEST_REQUIRE(original_groups[0].size() == 1);
  BOOST_TEST_REQUIRE(original_groups[1].size() == 2);
  BOOST_TEST(original_groups[2].empty());

  BOOST_TEST(original_groups[0][0] == entries[1].get());
  BOOST_TEST(original_groups[1][0] == entries[0].get());
  BOOST_TEST(original_groups[1][1] == entries[2].get());

  // Check if a range based loop, which uses begin() and end(), yields the
  // entries.
  size_t index = 0;
  for (const DeconvolutionTableEntry& entry : table) {
    BOOST_TEST(&entry == entries[index].get());
    ++index;
  }

  // Finally, check Front() and Size().
  BOOST_TEST(&table.Front() == entries.front().get());
  BOOST_TEST(table.Size() == entries.size());
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace radler