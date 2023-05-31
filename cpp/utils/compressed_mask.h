// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_UTILS_COMPRESSED_MASK_H_
#define RADLER_UTILS_COMPRESSED_MASK_H_

#include <cassert>
#include <string>
#include <vector>

#include "compressed_mask_data.h"

namespace radler::utils {

/**
 * A run-length encoding compressed image of boolean values. The class
 * stores counts of equivalent values. If the count is smaller than 255,
 * the count is stored as a uint8_t, if the count is smaller than 65536
 * it is stored as a uint16_t, otherwise it is stored as a uint64_t.
 *
 * Because a boolean also takes one byte, worst case (a checker-board
 * image) this memory footprint of this class is of the same size as a boolean
 * array, but because masks are commonly extremely sparse, the memory saving is
 * normally considerable.
 *
 * Because auto-masking with multi-scale enabled may require on the order of 10
 * such masks at full image size, the memory requirement this is considerable
 * when the data would be stored uncompressed. This class is aimed to solve
 * this.
 */
class CompressedMask : private implementation::CompressedMaskData {
 public:
  /**
   * Construct a mask for which all values are initialized to false.
   */
  CompressedMask(size_t width, size_t height) : width_(width), height_(height) {
    if (width > 0 && height > 0) PushBackCount(width * height);
  }

  size_t Width() const { return width_; }
  size_t Height() const { return height_; }

  /**
   * Obtain memory footprint of this class.
   */
  size_t CompressedSize() const { return sizeof(first_value_) + Data().size(); }

  /**
   * Uncompress the full data.
   * @param destination Allocation of at least Width() * Height() boolean values
   * where the data is stored.
   */
  void Get(bool* destination) const;

  /**
   * Replace the full data by compressing the specified data.
   */
  void Set(const bool* new_data);

  /**
   * Uncompress a rectangular data of the mask.
   * @param destination Allocation of at least @p width * @p height boolean
   * values where the data is stored.
   */
  void GetBox(bool* destination, size_t x, size_t y, size_t width,
              size_t height) const;

  /**
   * Replace a rectangular part of the data by compressing and merging the
   * specified data. The rows of the data are "on the fly" merged with the
   * compressed data, meaning that the full data is not completely uncompressed.
   */
  void SetBox(const bool* source, size_t x, size_t y, size_t width,
              size_t height);

  /**
   * Number of samples stored. It returns Width() * Height().
   * This function is aimed at testing class consistency.
   */
  size_t SumCount() const { return CompressedMaskData::SumCount(); }

  /**
   * Returns a string with a grid of "X" and "." values that represent the mask.
   */
  std::string ToString() const {
    return CompressedMaskData::ToString(first_value_, width_);
  }

 private:
  size_t width_;
  size_t height_;
  bool first_value_ = false;
};

}  // namespace radler::utils
#endif
