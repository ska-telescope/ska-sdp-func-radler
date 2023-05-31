// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_UTILS_COMPRESSED_MASK_DATA_H_
#define RADLER_UTILS_COMPRESSED_MASK_DATA_H_

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace radler::utils::implementation {

/**
 * Helper class for @ref CompressedData. This class holds the compressed
 * data counts, and provides low-level operations on this count. This class
 * makes it possible to test these low-level operations while the interface
 * CompressedData can remain simple.
 */
class CompressedMaskData {
 public:
  using iterator = std::vector<unsigned char>::iterator;
  using const_iterator = std::vector<unsigned char>::const_iterator;

  CompressedMaskData() noexcept = default;
  CompressedMaskData(CompressedMaskData&&) noexcept = default;
  CompressedMaskData(std::vector<unsigned char>&& data)
      : data_(std::move(data)) {}
  CompressedMaskData& operator=(CompressedMaskData&&) noexcept = default;

  const std::vector<unsigned char>& Data() const { return data_; }
  const_iterator begin() const { return data_.begin(); }
  iterator begin() { return data_.begin(); }
  const_iterator end() const { return data_.end(); }
  iterator end() { return data_.end(); }

  void Clear() { data_.clear(); }

  /** Sum of all compressed values, i.e., size of the uncompressed data. */
  size_t SumCount() const {
    size_t sum = 0;
    const_iterator iterator = data_.begin();
    while (iterator != data_.end()) {
      sum += GetCount(iterator);
    }
    return sum;
  }

  /**
   * Add a given number of samples with constant value add the end of the
   * compressed data. Values smaller than 255 take up 1 byte, whereas values >=
   * 256 take up 9 bytes (1 byte of value zero to indicate this is a long value,
   * then 8 bytes of data).
   * @param count should be larger than zero.
   */
  void PushBackCount(uint64_t count) {
    assert(count != 0);
    if (count < 255)
      data_.emplace_back(count);
    else if (count < 65536) {
      data_.emplace_back(255);
      data_.emplace_back(count % 256u);
      data_.emplace_back(count / 256u);
    } else {
      data_.emplace_back(0);
      const char* count_data = reinterpret_cast<char*>(&count);
      for (size_t i = 0; i != 8; ++i) data_.emplace_back(count_data[i]);
    }
  }

  /**
   * Compress a sequence of values and push them to the back of the data array.
   * @param [in] size Number of samples to compress in sequence
   * @param [in,out] sequence The data to compress of the specified size. The
   * pointer is moved to the end of the sequence.
   * @param [in,out] residual_count The number of samples with value
   * 'current_value' that needs to be added to the next pushed value.
   * @param [in,out] current_value The value associated to the next pushed count
   * (so the negation of the value associated to the previous pushed count).
   */
  void PushBackSequence(size_t size, const bool*& sequence,
                        size_t& residual_count, bool& current_value) {
    for (size_t i = 0; i != size; ++i) {
      if (*sequence != current_value) {
        PushBackCount(residual_count);
        residual_count = 0;
      }
      current_value = *sequence;
      ++residual_count;
      ++sequence;
    }
  }

  /**
   * Add a sequence from one compressed mask to this compressed mask.
   * @param [in,out] residual_count number of samples still available from
   * the source mask until the next read is required.
   * @return The number of samples with value 'current_value' that needs
   * to be added to the next pushed value.
   */
  size_t PushBackCompressedSequence(
      size_t size, const CompressedMaskData& source,
      std::vector<unsigned char>::const_iterator& source_iterator,
      size_t& residual_count, bool& current_value) {
    while (residual_count < size) {
      size -= residual_count;
      PushBackCount(residual_count);
      current_value = !current_value;
      // because we are inside while(residual_count < size), the iterator can
      // never be at the end, because then we would be asked more values than
      // available (which is considered undefined behaviour).
      residual_count = source.GetCount(source_iterator);
    }
    residual_count -= size;
    return size;
  }

  /**
   * Move the iterator one count value forward.
   */
  void SkipCount(std::vector<unsigned char>::const_iterator& iter) const {
    assert(iter != data_.end());
    const unsigned char value = *iter;
    ++iter;
    if (value == 255) {
      iter += 2;
    } else if (value == 0) {
      iter += 8;
    }
  }

  /**
   * Decode the count value of the iterator and move the iterator forward.
   * Behaviour is undefined when iter is at end.
   * @returns the decoded value (> 0).
   */
  uint64_t GetCount(std::vector<unsigned char>::const_iterator& iter) const {
    assert(iter != data_.end());
    const unsigned char value = *iter;
    ++iter;
    if (value == 0) {
      uint64_t u64_value;
      // The uint64_t is not stored aligned inside the vector, so it can not be
      // directly copied. Instead, copy it byte by byte.
      char* value_ptr = reinterpret_cast<char*>(&u64_value);
      std::copy_n(iter, 8, value_ptr);
      iter += 8;
      return u64_value;
    } else if (value == 255) {
      uint16_t u16_value = *iter;
      ++iter;
      u16_value += *iter << 8;
      ++iter;
      return u16_value;
    } else {
      return value;
    }
  }

  /**
   * Move iterator and update counters to skip a given number of uncompressed
   * samples.
   * @param [in] offset amount of uncompressed samples to skip
   * @param [in,out] iterator iterator into data pointing to the next compressed
   * count that needs to be read, or end when all values have been read.
   * @param [in,out] residual_count number of samples of the current count
   * available from the last read count, until the next change of value.
   * @param [in,out] value the value of the current serie of values. The next
   * uncompressed value has this value. Everytime a new count is read and the
   * iterator is increased, the value is negated.
   */
  void MoveIterator(size_t offset,
                    std::vector<unsigned char>::const_iterator& iterator,
                    size_t& residual_count, bool& value) const {
    while (offset >= residual_count) {
      offset -= residual_count;
      value = !value;
      if (iterator == data_.end()) {
        assert(offset == 0);
        residual_count = 0;
        break;
      } else {
        residual_count = GetCount(iterator);
      }
    }
    residual_count -= offset;
  }

  /**
   * Similar as @ref MoveIterator(), but will leave the iterator before the last
   * compressed value. On input, iterator should also point before the count
   * that is already in residual_count, i.e. it should lag one value behind
   * compared to @ref MoveIterator. Also, unlike MoveIterator, it will return
   * residual_count = 0 (and not read the next value already) in case offset
   * falls exactly after a compressed count value.
   */
  void MoveIteratorBefore(size_t offset,
                          std::vector<unsigned char>::const_iterator& iterator,
                          size_t& residual_count, bool& value) const {
    std::vector<unsigned char>::const_iterator next_iterator = iterator;
    SkipCount(next_iterator);

    while (offset > residual_count) {
      offset -= residual_count;
      value = !value;
      iterator = next_iterator;
      residual_count = GetCount(next_iterator);
    }
    residual_count -= offset;
  }

  /**
   * Decode a consecutive sequence of values and stop after a given number of
   * samples was decoded.
   * @param [in] size The number of uncompressed values to extract.
   * @param [in,out] destination Place for storing @c size values. The pointer
   * is moved to point the end of the sequence.
   * @param [in,out] iterator Position for the next read.
   * @param [in,out] residual_count How many values are still to be read with
   * the given value.
   * @param [in,out] value Value of the current read sequence.
   */
  void ExtractSequence(size_t size, bool*& destination,
                       std::vector<unsigned char>::const_iterator& iterator,
                       size_t& residual_count, bool& value) const {
    while (residual_count < size) {
      destination = std::fill_n(destination, residual_count, value);
      value = !value;
      size -= residual_count;
      residual_count = GetCount(iterator);
    }
    destination = std::fill_n(destination, size, value);
    residual_count -= size;
  }

  std::string ToString(bool first_value, size_t width) const;

 private:
  std::vector<unsigned char> data_;
};

}  // namespace radler::utils::implementation

#endif
