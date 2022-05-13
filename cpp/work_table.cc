// SPDX-License-Identifier: LGPL-3.0-only

#include "work_table.h"

#include <algorithm>
#include <cassert>

namespace radler {

WorkTable::WorkTable(int n_original_groups, int n_deconvolution_groups,
                     std::size_t channel_index_offset)
    : entries_(),
      channel_index_offset_(channel_index_offset),
      original_groups_(std::max(n_original_groups, 1)),
      deconvolution_groups_(
          (n_deconvolution_groups <= 0)
              ? original_groups_.size()
              : std::min(original_groups_.size(),
                         static_cast<size_t>(n_deconvolution_groups))) {
  // Create an entry in deconvolution_groups for each original group.
  for (int i = 0; i < n_original_groups; ++i) {
    int deconvolution_index =
        i * deconvolution_groups_.size() / n_original_groups;
    deconvolution_groups_[deconvolution_index].push_back(i);
  }
}

void WorkTable::AddEntry(std::unique_ptr<WorkTableEntry> entry) {
  const size_t original_channel_index = entry->original_channel_index;
  assert(original_channel_index < original_groups_.size());

  entry->index = entries_.size();
  entries_.push_back(std::move(entry));

  original_groups_[original_channel_index].push_back(entries_.back().get());
}
}  // namespace radler
