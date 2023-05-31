// SPDX-License-Identifier: LGPL-3.0-only

#include "utils/compressed_mask.h"

#include <algorithm>

#include <boost/test/unit_test.hpp>

namespace radler {

using utils::implementation::CompressedMaskData;

BOOST_AUTO_TEST_SUITE(compressed_mask_data)

BOOST_AUTO_TEST_CASE(construction) {
  CompressedMaskData data1;
  BOOST_CHECK(data1.Data().empty());
  data1.PushBackCount(1);
  data1.PushBackCount(2);

  std::vector<unsigned char> input{3, 4};
  CompressedMaskData data3(std::move(input));
  BOOST_REQUIRE_EQUAL(data3.Data().size(), 2);
  BOOST_CHECK_EQUAL(data3.Data()[0], 3);
  BOOST_CHECK_EQUAL(data3.Data()[1], 4);
}

BOOST_AUTO_TEST_CASE(clear) {
  CompressedMaskData data({3, 4, 0, 1, 0, 0, 0, 0, 0, 0});
  data.Clear();
  BOOST_CHECK(data.Data().empty());
}

BOOST_AUTO_TEST_CASE(sum_count) {
  BOOST_CHECK_EQUAL(CompressedMaskData().SumCount(), 0);
  CompressedMaskData data({3, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  BOOST_CHECK_EQUAL(data.SumCount(), 3 + 4 + 256);
}

BOOST_AUTO_TEST_CASE(push_back_count) {
  CompressedMaskData data;
  data.PushBackCount(1);
  BOOST_REQUIRE_EQUAL(data.Data().size(), 1);
  BOOST_CHECK_EQUAL(data.Data()[0], 1);
  data.PushBackCount(255);
  BOOST_REQUIRE_EQUAL(data.Data().size(), 4);
  BOOST_CHECK_EQUAL(data.SumCount(), 256);
  data.PushBackCount(256);
  BOOST_REQUIRE_EQUAL(data.Data().size(), 7);
  BOOST_CHECK_EQUAL(data.SumCount(), 512);
  data.PushBackCount(10000000);
  BOOST_REQUIRE_EQUAL(data.Data().size(), 16);
  BOOST_CHECK_EQUAL(data.SumCount(), 10000512);
}

BOOST_AUTO_TEST_CASE(push_back_sequence) {
  const std::array<bool, 7> input = {false, false, true, true,
                                     false, false, false};
  const bool* input_pointer = input.data();
  CompressedMaskData data;
  size_t residual_count = 0;
  bool current_value = false;

  data.PushBackSequence(1, input_pointer, residual_count, current_value);
  BOOST_CHECK_EQUAL(current_value, false);
  BOOST_CHECK_EQUAL(input_pointer - input.data(), 1);
  BOOST_CHECK_EQUAL(residual_count, 1);
  BOOST_CHECK(data.Data().empty());

  data.PushBackSequence(3, input_pointer, residual_count, current_value);
  BOOST_CHECK_EQUAL(current_value, true);
  BOOST_CHECK_EQUAL(input_pointer - input.data(), 4);
  BOOST_CHECK_EQUAL(residual_count, 2);
  BOOST_REQUIRE_EQUAL(data.ToString(false, 2), "..\n");

  data.PushBackSequence(3, input_pointer, residual_count, current_value);
  BOOST_CHECK_EQUAL(current_value, false);
  BOOST_CHECK_EQUAL(input_pointer - input.data(), 7);
  BOOST_CHECK_EQUAL(residual_count, 3);
  BOOST_REQUIRE_EQUAL(data.ToString(false, 4), "..XX\n");
}

BOOST_AUTO_TEST_CASE(push_back_compressed_sequence) {
  const CompressedMaskData input({3, 1, 4});
  CompressedMaskData::const_iterator iterator = input.Data().begin();
  size_t residual_count = input.GetCount(iterator);
  bool current_value = false;
  CompressedMaskData destination;

  size_t write_count = destination.PushBackCompressedSequence(
      1, input, iterator, residual_count, current_value);
  // Because the first sequence value '3' is not fully processed yet, the
  // destination shouldn't be changed. We've processed 1 uncompressed value,
  // which needs to be written later, so write_count should be 1. The iterator
  // is on the next-to-be-read value. From the first value (3), we've processed
  // one value, so residual_value should be 2.
  BOOST_CHECK_EQUAL(write_count, 1);
  BOOST_CHECK_EQUAL(iterator - input.Data().begin(), 1);
  BOOST_CHECK_EQUAL(residual_count, 2);
  BOOST_CHECK_EQUAL(current_value, false);
  BOOST_CHECK(destination.Data().empty());

  iterator = input.Data().begin();
  residual_count = input.GetCount(iterator);
  current_value = false;

  // This will process values '3' and '1' completely. However, the value '1'
  // should not have been written yet because future samples of the same value
  // should still be added to it. Therefore, only the '3' should have been
  // written to destination. There's one residual write left from the '1' that
  // wasn't written yet and we've fully processed the '1' and are ready to read
  // the next value (the '4') in a next read, which is indicated by a
  // residual_count of zero.
  write_count = destination.PushBackCompressedSequence(
      4, input, iterator, residual_count, current_value);
  BOOST_CHECK_EQUAL(write_count, 1);
  BOOST_CHECK_EQUAL(iterator - input.Data().begin(), 2);
  BOOST_CHECK_EQUAL(residual_count, 0);
  BOOST_CHECK_EQUAL(current_value, true);
  BOOST_CHECK_EQUAL(destination.ToString(false, 10), "...");

  // We must end on a full sequence for PushBackCompressedSequence, so
  // finish the current sequence:
  destination.PushBackCount(1);
  BOOST_CHECK_EQUAL(destination.ToString(false, 10), "...X");
  residual_count = input.GetCount(iterator);
  current_value = !current_value;
  // This tests a similar case as the previous call, but in this case reading
  // the last compressed value is tested (which triggers slightly different
  // code).
  write_count = destination.PushBackCompressedSequence(
      4, input, iterator, residual_count, current_value);
  BOOST_CHECK_EQUAL(write_count, 4);
  BOOST_CHECK_EQUAL(iterator - input.Data().begin(), 3);
  BOOST_CHECK_EQUAL(residual_count, 0);
  // current_value isn't strictly defined (there are no more values).
  BOOST_CHECK_EQUAL(destination.ToString(false, 10), "...X");
}

BOOST_AUTO_TEST_CASE(get_push_back_and_skip) {
  const std::array<uint64_t, 8> values{1,     255,   256,     10000,
                                       65535, 65536, 1000000, 858417};

  CompressedMaskData data;
  for (uint64_t value : values) {
    data.PushBackCount(value);
  }
  BOOST_CHECK_EQUAL(data.SumCount(), 2000000);

  CompressedMaskData::const_iterator iterator = data.begin();
  for (uint64_t value : values) {
    BOOST_CHECK_EQUAL(data.GetCount(iterator), value);
  }
  BOOST_CHECK(iterator == data.end());

  iterator = data.begin();
  for (size_t i = 1; i != values.size(); ++i) {
    data.SkipCount(iterator);
    CompressedMaskData::const_iterator temp_iterator = iterator;
    if (temp_iterator != data.end())
      BOOST_CHECK_EQUAL(values[i], data.GetCount(temp_iterator));
  }
  data.SkipCount(iterator);
  BOOST_CHECK(iterator == data.end());
}

BOOST_AUTO_TEST_CASE(extract_sequence) {
  const CompressedMaskData data({3, 7, 15});

  std::array<bool, 11> result;
  std::fill(result.begin(), result.end(), false);
  CompressedMaskData::const_iterator input_iterator = data.begin();
  bool* result_pointer = result.data();
  size_t residual_count = 0;
  bool current_value = false;
  data.ExtractSequence(0, result_pointer, input_iterator, residual_count,
                       current_value);
  BOOST_CHECK_EQUAL(input_iterator - data.begin(), 0);
  BOOST_CHECK(!current_value);
  BOOST_CHECK_EQUAL(residual_count, 0);
  BOOST_CHECK_EQUAL(result_pointer, result.data());
  BOOST_CHECK(!result[0]);  // Should be unchanged

  data.ExtractSequence(11, result_pointer, input_iterator, residual_count,
                       current_value);
  BOOST_CHECK_EQUAL(input_iterator - data.begin(), 3);
  BOOST_CHECK(current_value);
  BOOST_CHECK_EQUAL(residual_count, 14);
  BOOST_CHECK_EQUAL(result_pointer - result.data(), 11);
  const std::array<bool, 11> expected = {
      true, true, true, false, false, false, false, false, false, false, true};
  BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result_pointer,
                                expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(move_iterator_simple) {
  const CompressedMaskData data({1, 1, 1, 1, 1, 1});
  for (size_t start : {0, 2, 4, 5}) {
    for (size_t i = 0; i != 5 - start; ++i) {
      CompressedMaskData::const_iterator iterator =
          data.Data().begin() + start + 1;
      bool value = false;
      size_t residual_count = 1;
      data.MoveIterator(i, iterator, residual_count, value);
      BOOST_CHECK_EQUAL(iterator - data.Data().begin(), i + start + 1);
      BOOST_CHECK_EQUAL(value, i % 2 != 0);
      BOOST_CHECK_EQUAL(residual_count, 1);
    }
  }

  for (size_t start : {0, 2, 4, 5}) {
    CompressedMaskData::const_iterator iterator =
        data.Data().begin() + start + 1;
    bool value = false;
    size_t residual_count = 1;
    data.MoveIterator(6 - start, iterator, residual_count, value);
    BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 6);
    BOOST_CHECK_EQUAL(residual_count, 0);
  }
}

BOOST_AUTO_TEST_CASE(move_iterator_complex) {
  const CompressedMaskData data({3, 1, 5});
  CompressedMaskData::const_iterator iterator = data.Data().begin() + 1;
  bool value = false;
  size_t residual_count = data.Data()[0];

  for (size_t i = 0; i != 2; ++i) {
    data.MoveIterator(1, iterator, residual_count, value);
    BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 1);
    BOOST_CHECK_EQUAL(value, false);
    BOOST_CHECK_EQUAL(residual_count, 2 - i);
  }

  data.MoveIterator(1, iterator, residual_count, value);
  BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 2);
  BOOST_CHECK_EQUAL(value, true);
  BOOST_CHECK_EQUAL(residual_count, 1);

  for (size_t i = 0; i != 5; ++i) {
    data.MoveIterator(1, iterator, residual_count, value);
    BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 3);
    BOOST_CHECK_EQUAL(value, false);
    BOOST_CHECK_EQUAL(residual_count, 5 - i);
  }
}

BOOST_AUTO_TEST_CASE(move_iterator_jump) {
  const CompressedMaskData data({3, 1, 5, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  CompressedMaskData::const_iterator iterator = data.Data().begin() + 1;
  bool value = true;
  size_t residual_count = data.Data()[0];

  data.MoveIterator(3, iterator, residual_count, value);
  BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 2);
  BOOST_CHECK_EQUAL(value, false);
  BOOST_CHECK_EQUAL(residual_count, 1);

  data.MoveIterator(10, iterator, residual_count, value);
  BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 13);
  BOOST_CHECK_EQUAL(value, true);
  BOOST_CHECK_EQUAL(residual_count, 256 - 3);
}

BOOST_AUTO_TEST_CASE(move_iterator_before_simple) {
  const CompressedMaskData data({1, 1, 1, 1, 1, 1});
  for (size_t start : {0, 2, 4, 5}) {
    for (size_t i = 1; i < 5 - start; ++i) {
      CompressedMaskData::const_iterator iterator = data.Data().begin() + start;
      bool value = false;
      size_t residual_count = 1;
      data.MoveIteratorBefore(i, iterator, residual_count, value);
      BOOST_CHECK_EQUAL(iterator - data.Data().begin(), i + start - 1);
      BOOST_CHECK_EQUAL(value, i % 2 == 0);
      BOOST_CHECK_EQUAL(residual_count, 0);
    }
  }

  for (size_t start : {0, 2, 4, 5}) {
    CompressedMaskData::const_iterator iterator = data.Data().begin() + start;
    bool value = false;
    size_t residual_count = 1;
    data.MoveIteratorBefore(6 - start, iterator, residual_count, value);
    BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 5);
    BOOST_CHECK_EQUAL(residual_count, 0);
  }
}

BOOST_AUTO_TEST_CASE(move_iterator_before_bug) {
  // This tests a specific case that triggered a bug in a previous version of
  // MoveIteratorBefore
  const CompressedMaskData data({4, 2, 1, 2, 6});
  CompressedMaskData::const_iterator iterator = data.Data().begin();
  bool value = false;
  size_t residual_count = data.Data()[0];
  data.MoveIteratorBefore(4, iterator, residual_count, value);
  BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 0);
  BOOST_CHECK_EQUAL(value, false);
  BOOST_CHECK_EQUAL(residual_count, 0);
}

BOOST_AUTO_TEST_CASE(move_iterator_before_complex) {
  const CompressedMaskData data({3, 1, 5});
  CompressedMaskData::const_iterator iterator = data.Data().begin();
  bool value = false;
  size_t residual_count = data.Data()[0];

  for (size_t i = 0; i != 3; ++i) {
    data.MoveIteratorBefore(1, iterator, residual_count, value);
    BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 0);
    BOOST_CHECK_EQUAL(value, false);
    BOOST_CHECK_EQUAL(residual_count, 2 - i);
  }

  data.MoveIteratorBefore(1, iterator, residual_count, value);
  BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 1);
  BOOST_CHECK_EQUAL(value, true);
  BOOST_CHECK_EQUAL(residual_count, 0);

  for (size_t i = 0; i != 5; ++i) {
    data.MoveIteratorBefore(1, iterator, residual_count, value);
    BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 2);
    BOOST_CHECK_EQUAL(value, false);
    BOOST_CHECK_EQUAL(residual_count, 4 - i);
  }
}

BOOST_AUTO_TEST_CASE(move_iterator_before_jump) {
  const CompressedMaskData data({3, 1, 5, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  CompressedMaskData::const_iterator iterator = data.Data().begin();
  bool value = true;
  size_t residual_count = data.Data()[0];

  data.MoveIteratorBefore(3, iterator, residual_count, value);
  BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 0);
  BOOST_CHECK_EQUAL(value, true);
  BOOST_CHECK_EQUAL(residual_count, 0);

  data.MoveIteratorBefore(10, iterator, residual_count, value);
  BOOST_CHECK_EQUAL(iterator - data.Data().begin(), 4);
  BOOST_CHECK_EQUAL(value, true);
  BOOST_CHECK_EQUAL(residual_count, 256 - 3);
}

BOOST_AUTO_TEST_CASE(to_string) {
  CompressedMaskData data;
  BOOST_CHECK_EQUAL(data.ToString(true, 0), "");

  data = CompressedMaskData({20});
  BOOST_CHECK_EQUAL(data.ToString(false, 4), "....\n....\n....\n....\n....\n");

  data = CompressedMaskData({2, 2, 2, 2, 1});
  BOOST_CHECK_EQUAL(data.ToString(true, 3), "XX.\n.XX\n..X\n");
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace radler
