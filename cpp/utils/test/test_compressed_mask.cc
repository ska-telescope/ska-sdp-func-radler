// SPDX-License-Identifier: LGPL-3.0-only

#include "utils/compressed_mask.h"

#include <algorithm>

#include <boost/test/unit_test.hpp>

namespace radler {

using utils::CompressedMask;

BOOST_AUTO_TEST_SUITE(compressed_mask)

BOOST_AUTO_TEST_CASE(construction) {
  const CompressedMask mask(3, 4);
  BOOST_CHECK_EQUAL(mask.Width(), 3);
  BOOST_CHECK_EQUAL(mask.Height(), 4);
  BOOST_CHECK_EQUAL(mask.CompressedSize(),
                    sizeof(bool) + 1);  // 1 count value of '12'
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  std::array<bool, 12> data;
  mask.Get(data.data());
  std::array<bool, 12> reference;
  std::fill_n(reference.begin(), 12, false);
  BOOST_CHECK_EQUAL_COLLECTIONS(reference.begin(), reference.end(),
                                data.begin(), data.end());
}

BOOST_AUTO_TEST_CASE(empty_mask) {
  CompressedMask mask(3, 0);
  BOOST_CHECK_EQUAL(mask.Width(), 3);
  BOOST_CHECK_EQUAL(mask.Height(), 0);
  BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool));
  BOOST_CHECK_EQUAL(mask.SumCount(), 0);
  BOOST_CHECK_NO_THROW(mask.Get(nullptr));
  BOOST_CHECK_NO_THROW(mask.Set(nullptr));
  BOOST_CHECK_NO_THROW(mask.GetBox(nullptr, 0, 0, 0, 0));
  BOOST_CHECK_NO_THROW(mask.SetBox(nullptr, 0, 0, 0, 0));
}

BOOST_AUTO_TEST_CASE(construct_large) {
  const CompressedMask mask(300, 500);
  BOOST_CHECK_EQUAL(mask.Width(), 300);
  BOOST_CHECK_EQUAL(mask.Height(), 500);
  // Compressed should use a 64-bit value with a 0 prefix for a count of 150000.
  BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool) + 1 + 8);
}

BOOST_AUTO_TEST_CASE(compress) {
  CompressedMask mask(3, 4);
  const std::array<bool, 12> input = {false, true, true, false, false, false,
                                      true,  true, true, true,  false, false};
  mask.Set(input.data());
  BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool) + 5);
  std::array<bool, 12> result;
  std::fill(result.begin(), result.end(), false);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.Get(result.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(), result.begin(),
                                result.end());
}

BOOST_AUTO_TEST_CASE(get_box_small) {
  CompressedMask mask(4, 4);
  const std::array<bool, 16> input = {false, true,  true,  false,  // 1
                                      false, false, true,  true,   // 2
                                      true,  true,  false, false,  // 3
                                      false, false, false, true};
  mask.Set(input.data());
  BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool) + 6);
  std::array<bool, 6> result;
  std::fill(result.begin(), result.end(), false);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.GetBox(result.data(), 1, 1, 2, 3);
  const std::array<bool, 6> expected = {false, true, true, false, false, false};
  BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                result.begin(), result.end());
}

BOOST_AUTO_TEST_CASE(get_box_with_full_width) {
  CompressedMask mask(4, 5);
  const std::array<bool, 20> input = {
      false, true,  true,  false,  // 1
      false, false, true,  true,   // 2
      true,  true,  false, false,  // 3
      false, false, false, true,   // 4
      true,  true,  true,  true    // 5
  };
  mask.Set(input.data());
  std::array<bool, 16> result;
  std::fill(result.begin(), result.end(), false);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.GetBox(result.data(), 0, 1, 4, 4);
  BOOST_CHECK_EQUAL_COLLECTIONS(input.begin() + 4, input.end(), result.begin(),
                                result.end());
}

BOOST_AUTO_TEST_CASE(get_box_with_full_mask) {
  CompressedMask mask(3, 3);
  const std::array<bool, 9> input = {
      true,  true,  true,   // 1
      true,  false, false,  // 2
      false, false, true    // 3
  };
  mask.Set(input.data());
  std::array<bool, 9> result;
  std::fill(result.begin(), result.end(), false);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.GetBox(result.data(), 0, 0, 3, 3);
  BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(), result.begin(),
                                result.end());
}

BOOST_AUTO_TEST_CASE(set_box_side) {
  CompressedMask mask(3, 5);
  for (bool value : {false, true, true, false}) {
    const std::array<bool, 4> input = {value, value, value, value};
    mask.SetBox(input.data(), 1, 1, 2, 2);
    BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
    std::array<bool, 15> result;
    mask.Get(result.data());
    const std::array<bool, 15> expected = {
        false, false, false,  // 1
        false, value, value,  // 2
        false, value, value,  // 3
        false, false, false,  // 4
        false, false, false   // 5
    };
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                  result.begin(), result.end());
    BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool) + (value ? 5 : 1));
  }
}

BOOST_AUTO_TEST_CASE(set_box_mid) {
  CompressedMask mask(3, 3);
  for (bool value : {false, true, true, false}) {
    const std::array<bool, 1> input = {value};
    mask.SetBox(input.data(), 1, 1, 1, 1);
    BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
    std::array<bool, 9> result;
    mask.Get(result.data());
    const std::array<bool, 9> expected = {
        false, false, false,  // 1
        false, value, false,  // 2
        false, false, false   // 3
    };
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                  result.begin(), result.end());
    BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool) + (value ? 3 : 1));
  }
}

BOOST_AUTO_TEST_CASE(set_box_with_first_value) {
  CompressedMask mask(2, 2);
  for (bool value : {false, true, true, false}) {
    const std::array<bool, 2> input = {true, value};
    mask.SetBox(input.data(), 0, 0, 1, 2);
    BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
    std::array<bool, 4> result;
    mask.Get(result.data());
    const std::array<bool, 4> expected = {
        true, false,  // 1
        value, false  // 2
    };
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                  result.begin(), result.end());
  }
}

BOOST_AUTO_TEST_CASE(set_box_full) {
  CompressedMask mask(3, 3);
  std::array<bool, 9> input;
  std::fill(input.begin(), input.end(), true);
  mask.SetBox(input.data(), 0, 0, 3, 3);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  std::array<bool, 9> result;
  mask.Get(result.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(), result.begin(),
                                result.end());
  BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool) + 1);
}

BOOST_AUTO_TEST_CASE(set_box_checkers_small) {
  CompressedMask mask(3, 2);
  std::array<bool, 6> input{true,  false, true,  // 1
                            false, true,  false};
  mask.SetBox(input.data(), 0, 0, 3, 2);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());

  std::array<bool, 6> result;
  mask.Get(result.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(), result.begin(),
                                result.end());

  mask.SetBox(input.data(), 1, 1, 2, 1);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.Get(result.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(), result.begin(),
                                result.end());
}

BOOST_AUTO_TEST_CASE(set_box_checkers_big) {
  CompressedMask mask(5, 5);
  std::array<bool, 25> input{true};
  for (size_t i = 1; i != 25; i += 2) {
    input[i] = false;
    input[i + 1] = true;
  }
  mask.SetBox(input.data(), 0, 0, 5, 5);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());

  std::array<bool, 25> result;
  mask.Get(result.data());
  BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(), result.begin(),
                                result.end());
  BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool) + 25);

  mask.SetBox(input.data(), 1, 1, 2, 2);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.Get(result.data());
  constexpr std::array<bool, 25> expected_a = {
      true,  false, true,  false, true,   // 1
      false, true,  false, true,  false,  // 2
      true,  true,  false, false, true,   // 3
      false, true,  false, true,  false,  // 4
      true,  false, true,  false, true    // 5
  };
  BOOST_CHECK_EQUAL_COLLECTIONS(expected_a.begin(), expected_a.end(),
                                result.begin(), result.end());

  mask.SetBox(input.data(), 3, 2, 2, 2);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.Get(result.data());
  constexpr std::array<bool, 25> expected_b = {
      true,  false, true,  false, true,   // 1
      false, true,  false, true,  false,  // 2
      true,  true,  false, true,  false,  // 3
      false, true,  false, true,  false,  // 4
      true,  false, true,  false, true    // 5
  };
  BOOST_CHECK_EQUAL_COLLECTIONS(expected_b.begin(), expected_b.end(),
                                result.begin(), result.end());

  std::fill(input.begin(), input.end(), true);
  mask.SetBox(input.data(), 2, 0, 2, 5);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.Get(result.data());
  constexpr std::array<bool, 25> expected_c = {
      true,  false, true, true, true,   // 1
      false, true,  true, true, false,  // 2
      true,  true,  true, true, false,  // 3
      false, true,  true, true, false,  // 4
      true,  false, true, true, true    // 5
  };
  BOOST_CHECK_EQUAL_COLLECTIONS(expected_c.begin(), expected_c.end(),
                                result.begin(), result.end());

  std::fill(input.begin(), input.end(), true);
  mask.SetBox(input.data(), 0, 3, 5, 2);
  BOOST_CHECK_EQUAL(mask.SumCount(), mask.Width() * mask.Height());
  mask.Get(result.data());
  constexpr std::array<bool, 25> expected_d = {
      true,  false, true, true, true,   // 1
      false, true,  true, true, false,  // 2
      true,  true,  true, true, false,  // 3
      true,  true,  true, true, true,   // 4
      true,  true,  true, true, true    // 5
  };
  BOOST_CHECK_EQUAL_COLLECTIONS(expected_d.begin(), expected_d.end(),
                                result.begin(), result.end());
  BOOST_CHECK_EQUAL(mask.CompressedSize(), sizeof(bool) + 9);
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace radler
