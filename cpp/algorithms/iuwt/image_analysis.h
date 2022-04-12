// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_ALGORITHMS_IUWT_IMAGE_ANALYSIS_H_
#define RADLER_ALGORITHMS_IUWT_IMAGE_ANALYSIS_H_

#include "algorithms/iuwt/iuwt_decomposition.h"

namespace radler::algorithms::iuwt {

class ImageAnalysis {
 public:
  struct Component {
    Component(size_t _x, size_t _y, int _scale) : x(_x), y(_y), scale(_scale) {}

    std::string ToString() const {
      std::ostringstream str;
      str << x << ',' << y << ", scale " << scale;
      return str.str();
    }

    size_t x, y;
    int scale;
  };

  struct Component2D {
    Component2D(size_t _x, size_t _y) : x(_x), y(_y) {}

    std::string ToString() const {
      std::ostringstream str;
      str << x << ',' << y;
      return str.str();
    }

    size_t x, y;
  };

  static void SelectStructures(const IUWTDecomposition& iuwt, IUWTMask& mask,
                               const aocommon::UVector<float>& thresholds,
                               size_t minScale, size_t endScale,
                               float cleanBorder, const bool* priorMask,
                               size_t& areaSize);

  static bool IsHighestOnScale0(const IUWTDecomposition& iuwt,
                                IUWTMask& markedMask, size_t& x, size_t& y,
                                size_t endScale, float& highestScale0);

  static void Floodfill(const IUWTDecomposition& iuwt, IUWTMask& mask,
                        const aocommon::UVector<float>& thresholds,
                        size_t minScale, size_t endScale,
                        const Component& component, float cleanBorder,
                        size_t& areaSize);

  static void MaskedFloodfill(const IUWTDecomposition& iuwt, IUWTMask& mask,
                              const aocommon::UVector<float>& thresholds,
                              size_t minScale, size_t endScale,
                              const Component& component, float cleanBorder,
                              const bool* priorMask, size_t& areaSize);

  static void FloodFill2D(const float* image, bool* mask, float threshold,
                          const Component2D& component, size_t width,
                          size_t height, size_t& areaSize);

  /**
   * Exactly like above, but now collecting the components in the
   * area vector, instead of returning just the area size.
   */
  static void FloodFill2D(const float* image, bool* mask, float threshold,
                          const Component2D& component, size_t width,
                          size_t height, std::vector<Component2D>& area);

 private:
  static bool exceedsThreshold(float val, float threshold) {
    if (threshold >= 0.0)
      return val > threshold;
    else
      return val < threshold || val > -threshold;
  }
  static bool exceedsThresholdAbs(float val, float threshold) {
    return std::fabs(val) > threshold;
  }
};
}  // namespace radler::algorithms::iuwt
#endif  // RADLER_ALGORITHMS_IUWT_IMAGE_ANALYSIS_H_
