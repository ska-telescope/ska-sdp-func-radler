// SPDX-License-Identifier: LGPL-3.0-only

#include "algorithms/iuwt/iuwt_mask.h"

#include <limits>

#include "algorithms/iuwt/iuwt_decomposition.h"

namespace radler::algorithms::iuwt {

std::string IuwtMask::Summary(const IuwtDecomposition& iuwt) const {
  std::ostringstream str;
  str << "IUWTMask with " << _masks.size()
      << " scale masks (iuwt: " << iuwt.Summary() << ")\n";
  for (size_t i = 0; i != _masks.size(); ++i) {
    double maxVal = std::numeric_limits<double>::lowest();
    double minVal = std::numeric_limits<double>::max();
    size_t count = 0;
    for (size_t j = 0; j != _masks[i].size(); ++j) {
      if (_masks[i][j]) {
        ++count;
        if (iuwt[i][j] > maxVal) maxVal = iuwt[i][j];
        if (iuwt[i][j] < minVal) minVal = iuwt[i][j];
      }
    }
    if (maxVal == std::numeric_limits<double>::lowest()) {
      maxVal = std::numeric_limits<double>::quiet_NaN();
      minVal = std::numeric_limits<double>::quiet_NaN();
    }
    str << "Scale " << i << ": " << count << " (" << minVal << " - " << maxVal
        << ")\n";
  }
  return str.str();
}
}  // namespace radler::algorithms::iuwt
