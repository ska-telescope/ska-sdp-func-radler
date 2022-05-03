// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_ALGORITHMS_THREADED_DECONVOLUTION_TOOLS_H_
#define RADLER_ALGORITHMS_THREADED_DECONVOLUTION_TOOLS_H_

#include <cmath>
#include <optional>
#include <thread>
#include <vector>

#include <aocommon/image.h>
#include <aocommon/lane.h>
#include <aocommon/uvector.h>

#include "algorithms/multiscale/multiscale_transforms.h"

namespace radler::algorithms {

class ThreadedDeconvolutionTools {
 public:
  explicit ThreadedDeconvolutionTools(size_t threadCount);
  ~ThreadedDeconvolutionTools();

  ThreadedDeconvolutionTools(const ThreadedDeconvolutionTools&) = default;
  ThreadedDeconvolutionTools(ThreadedDeconvolutionTools&&) = default;
  ThreadedDeconvolutionTools& operator=(const ThreadedDeconvolutionTools&) =
      default;
  ThreadedDeconvolutionTools& operator=(ThreadedDeconvolutionTools&&) = default;

  struct PeakData {
    std::optional<float> normalizedValue, unnormalizedValue;
    float rms;
    size_t x, y;
  };

  void SubtractImage(float* image, const aocommon::Image& psf, size_t x,
                     size_t y, float factor);

  void FindMultiScalePeak(
      multiscale::MultiScaleTransforms* msTransforms,
      const aocommon::Image& image, const aocommon::UVector<float>& scales,
      std::vector<PeakData>& results, bool allowNegativeComponents,
      const bool* mask, const std::vector<aocommon::UVector<bool>>& scaleMasks,
      float borderRatio, const aocommon::Image& rmsFactorImage,
      bool calculateRMS);

  static float RMS(const aocommon::Image& image, size_t n) {
    float result = 0.0;
    for (size_t i = 0; i != n; ++i) result += image[i] * image[i];
    return std::sqrt(result / float(n));
  }

 private:
  struct ThreadResult {};
  struct FindMultiScalePeakResult : public ThreadResult {
    std::optional<float> unnormalizedValue, normalizedValue;
    float rms;
    size_t x, y;
  };

  struct ThreadTask {
    virtual std::unique_ptr<ThreadResult> operator()() = 0;
    virtual ~ThreadTask() {}
  };
  struct SubtractionTask : public ThreadTask {
    virtual std::unique_ptr<ThreadResult> operator()();

    float* image;
    const aocommon::Image* psf;
    size_t x, y;
    float factor;
    size_t startY, endY;
  };

  struct FindMultiScalePeakTask : public ThreadTask {
    virtual std::unique_ptr<ThreadResult> operator()();

    multiscale::MultiScaleTransforms* msTransforms;
    aocommon::Image* image;
    aocommon::Image* scratch;
    float scale;
    bool allowNegativeComponents;
    const bool* mask;
    float borderRatio;
    bool calculateRMS;
    const aocommon::Image* rmsFactorImage;
  };

  std::vector<aocommon::Lane<std::unique_ptr<ThreadTask>>> _taskLanes;
  std::vector<aocommon::Lane<std::unique_ptr<ThreadResult>>> _resultLanes;
  size_t _threadCount;
  std::vector<std::thread> _threadGroup;

  void threadFunc(aocommon::Lane<std::unique_ptr<ThreadTask>>* taskLane,
                  aocommon::Lane<std::unique_ptr<ThreadResult>>* resultLane) {
    std::unique_ptr<ThreadTask> task;
    while (taskLane->read(task)) {
      std::unique_ptr<ThreadResult> result = (*task)();
      resultLane->write(std::move(result));
    }
  }
};
}  // namespace radler::algorithms
#endif
