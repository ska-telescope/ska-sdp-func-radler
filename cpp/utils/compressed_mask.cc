#include "compressed_mask.h"

#include <sstream>

namespace radler::utils {

std::string implementation::CompressedMaskData::ToString(bool first_value,
                                                         size_t width) const {
  std::ostringstream result;
  const_iterator i = data_.begin();
  size_t row = 0;
  bool value = first_value;
  while (i != data_.end()) {
    size_t count = GetCount(i);
    for (size_t j = 0; j != count; ++j) {
      result << (value ? 'X' : '.');
      ++row;
      if (row == width) {
        row = 0;
        result << '\n';
      }
    }
    value = !value;
  }
  return result.str();
}

void CompressedMask::Get(bool* destination) const {
  bool value = first_value_;
  const_iterator iterator = Data().begin();
  while (iterator != Data().end()) {
    size_t count = GetCount(iterator);  // Increases 'iterator'
    destination = std::fill_n(destination, count, value);
    value = !value;
  }
}

void CompressedMask::Set(const bool* new_data) {
  if (width_ == 0 || height_ == 0) return;

  Clear();
  first_value_ = new_data[0];
  bool current_value = first_value_;
  const bool* end = new_data + width_ * height_;
  size_t count = 1;
  ++new_data;
  while (new_data != end) {
    if (*new_data != current_value) {
      PushBackCount(count);
      count = 0;
      current_value = *new_data;
    }
    ++count;
    ++new_data;
  }
  PushBackCount(count);
}

void CompressedMask::GetBox(bool* destination, size_t x, size_t y, size_t width,
                            size_t height) const {
  assert(x + width <= width_);
  assert(y + height <= height_);
  if (width == 0 || height == 0) return;

  bool value = first_value_;
  const_iterator iterator = Data().begin();
  size_t count = GetCount(iterator);

  MoveIterator(x + y * width_, iterator, count, value);
  for (size_t current_line = 0; current_line + 1 != height; ++current_line) {
    ExtractSequence(width, destination, iterator, count, value);
    MoveIterator(width_ - width, iterator, count, value);
  }
  ExtractSequence(width, destination, iterator, count, value);
}

void CompressedMask::SetBox(const bool* source, size_t x, size_t y,
                            size_t width, size_t height) {
  assert(x + width <= width_);
  assert(y + height <= height_);
  if (width == 0 || height == 0) return;

  // STAGE 1: copy the unchanged values

  const_iterator read_iterator = Data().begin();
  // read_count is at all times the number of decompressed values from the 'old'
  // data that still need to be processed, and is associated with read_value.
  size_t read_count = GetCount(read_iterator);
  bool read_value = first_value_;
  read_iterator = Data().begin();
  // move to the position at which the changes start and copy all values. Move
  // one value before the last count, because this count might require a change.
  MoveIteratorBefore(x + y * width_, read_iterator, read_count, read_value);
  CompressedMaskData new_mask(
      std::vector<unsigned char>(Data().cbegin(), read_iterator));

  bool current_value = read_value;
  if (x == 0 && y == 0) {
    first_value_ = source[0];
    current_value = source[0];
  }
  // number of already processed samples that still need to be written to the
  // compressed values.
  size_t write_count = GetCount(read_iterator) - read_count;

  // STAGE 2: Compress/copy the data line by line. Each iteration consists of
  // first writing the changed value, and (if necessary, i.e. when width <
  // width_) finishes the line with the unchanged values from the old data.

  for (size_t j = 0; j != height - 1; ++j) {
    MoveIterator(width, read_iterator, read_count, read_value);
    new_mask.PushBackSequence(width, source, write_count, current_value);

    size_t next_line = width_ - width;
    if (current_value != read_value && next_line) {
      // If the value changes, we've finished this series and can directly copy
      // the remaining values.
      new_mask.PushBackCount(write_count);
      current_value = !current_value;
      write_count = new_mask.PushBackCompressedSequence(
          next_line, *this, read_iterator, read_count, read_value);
      current_value = read_value;
    } else if (read_count < next_line) {
      // The value does not change at the old-new boundary, but it does change
      // before the next line starts, so can be written.
      new_mask.PushBackCount(write_count + read_count);
      current_value = !current_value;
      next_line -= read_count;
      write_count += read_count;
      read_count = GetCount(read_iterator);
      read_value = !read_value;
      write_count = new_mask.PushBackCompressedSequence(
          next_line, *this, read_iterator, read_count, read_value);
      current_value = read_value;
    } else {
      // The value does not change (neither at the old-new boundary, nor before
      // the next line). In this case, we just need to update the counters so
      // that the constant values are written later.
      read_count -= next_line;
      write_count += next_line;
    }
  }

  // STAGE 3: Compress the last line of the changed data.

  MoveIterator(width, read_iterator, read_count, read_value);
  new_mask.PushBackSequence(width, source, write_count, current_value);

  // STAGE 4: Copy all remaining unchanged data.

  // Remaining number of values to write:
  size_t remaining = width_ * (height_ + 1 - y - height) - x - width;
  if (read_value != current_value) {
    // If the value changes on the new-old boundary, write the value up to the
    // boundary:
    new_mask.PushBackCount(write_count);
    write_count = 0;
    current_value = !current_value;
    if (remaining) {
      write_count = new_mask.PushBackCompressedSequence(
          remaining, *this, read_iterator, read_count, read_value);
    }
  } else {
    // If the value doesn't change, write up to where the value does change:
    new_mask.PushBackCount(write_count + read_count);
    remaining -= read_count;
    write_count = 0;
    current_value = !current_value;
    if (remaining) {
      read_count = GetCount(read_iterator);
      write_count = new_mask.PushBackCompressedSequence(
          remaining, *this, read_iterator, read_count, read_value);
    }
  }
  if (write_count) new_mask.PushBackCount(write_count);
  CompressedMaskData::operator=(std::move(new_mask));
}

}  // namespace radler::utils
