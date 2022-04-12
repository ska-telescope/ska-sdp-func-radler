// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_DECONVOLUTION_TABLE_H_
#define RADLER_DECONVOLUTION_TABLE_H_

#include "deconvolution_table_entry.h"

#include <functional>
#include <memory>
#include <vector>

namespace radler {
/**
 * The DeconvolutionTable contains DeconvolutionTableEntry's and groups entries
 * that have the same squaredDeconvolutionIndex.
 */
class DeconvolutionTable {
 public:
  using Entries = std::vector<std::unique_ptr<DeconvolutionTableEntry>>;
  using Group = std::vector<const DeconvolutionTableEntry*>;

  /**
   * Iterator-like class which (only) supports a range-based loop over entries.
   *
   * Dereferencing this class yields a reference to the actual object instead
   * of a reference to the pointer for the object.
   */
  class EntryIteratorLite {
    using BaseIterator = Entries::const_iterator;

   public:
    explicit EntryIteratorLite(BaseIterator base_iterator)
        : base_iterator_(base_iterator) {}

    const DeconvolutionTableEntry& operator*() const {
      return **base_iterator_;
    }
    EntryIteratorLite& operator++() {
      ++base_iterator_;
      return *this;
    }
    bool operator!=(const EntryIteratorLite& other) const {
      return base_iterator_ != other.base_iterator_;
    }
    bool operator==(const EntryIteratorLite& other) const {
      return base_iterator_ == other.base_iterator_;
    }

   private:
    BaseIterator base_iterator_;
  };

  /**
   * @brief Constructs a new DeconvolutionTable object.
   *
   * @param n_original_groups The number of original channel groups. When adding
   * entries, their original channel index must be less than the number of
   * original groups. If the value is zero or less, one group is used.
   * @param n_deconvolution_groups The number of deconvolution groups.
   * A deconvolution group consist of one or more channel groups, which are then
   * joinedly deconvolved.
   * If the value is zero or less, or larger than the number of original groups,
   * all channels are deconvolved separately.
   * @param channel_index_offset The index of the first channel in the caller.
   * Must be >= 0.
   */
  explicit DeconvolutionTable(int n_original_groups, int n_deconvolution_groups,
                              int channel_index_offset = 0);

  /**
   * @return The table entries, grouped by their original channel index.
   * @see AddEntry()
   */
  const std::vector<Group>& OriginalGroups() const { return original_groups_; }

  /**
   * @return The original group indices for each deconvolution group.
   */
  const std::vector<std::vector<int>>& DeconvolutionGroups() const {
    return deconvolution_groups_;
  }

  /**
   * Find the first group of original channels, given a deconvolution group
   * index.
   *
   * @param deconvolution_index Index for a deconvolution group. Must be less
   * than the number of deconvolution groups.
   * @return A reference to the first original group for the deconvolution
   * group.
   */
  const Group& FirstOriginalGroup(size_t deconvolution_index) const {
    return original_groups_[deconvolution_groups_[deconvolution_index].front()];
  }

  /**
   * begin() and end() allow writing range-based loops over all entries.
   * @{
   */
  EntryIteratorLite begin() const {
    return EntryIteratorLite(entries_.begin());
  }
  EntryIteratorLite end() const { return EntryIteratorLite(entries_.end()); }
  /** @} */

  /**
   * @brief Adds an entry to the table.
   *
   * The original channel index of the entry determines the original group for
   * the entry. It must be less than the number of original channel groups, as
   * given in the constructor.
   *
   * @param entry A new entry.
   */
  void AddEntry(std::unique_ptr<DeconvolutionTableEntry> entry);

  /**
   * @return A reference to the first entry.
   */
  const DeconvolutionTableEntry& Front() const { return *entries_.front(); }

  /**
   * @return The number of entries in the table.
   */
  size_t Size() const { return entries_.size(); }

  /**
   * @return The channel index offset, which was set in the constructor.
   */
  size_t GetChannelIndexOffset() const { return channel_index_offset_; }

 private:
  Entries entries_;

  /**
   * A user of the DeconvolutionTable may use different channel indices than
   * the DeconvolutionTable. This offset is the difference between those
   * indices.
   * For example, with three channels, the DeconvolutionTable indices are always
   * 0, 1, and 2. When the user indices are 4, 5, and 6, this offset will be 4.
   */
  const std::size_t channel_index_offset_;

  /**
   * An original group has entries with equal original channel indices.
   */
  std::vector<Group> original_groups_;

  /**
   * A deconvolution group consists of one or more original groups, which
   * are deconvolved together. Each entry contains the indices of the original
   * groups that are part of the deconvolution group.
   */
  std::vector<std::vector<int>> deconvolution_groups_;
};
}  // namespace radler

#endif
