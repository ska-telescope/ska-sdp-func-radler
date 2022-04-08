// Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "algorithms/parallel_deconvolution.h"

#include <aocommon/parallelfor.h>
#include <aocommon/units/fluxdensity.h>

#include <schaapcommon/fft/convolution.h>

#include "algorithms/multiscale_algorithm.h"
#include "math/dijkstrasplitter.h"

using aocommon::Image;
using aocommon::Logger;

namespace radler::algorithms {

ParallelDeconvolution::ParallelDeconvolution(
    const DeconvolutionSettings& deconvolutionSettings)
    : _horImages(0),
      _verImages(0),
      _settings(deconvolutionSettings),
      _allocator(nullptr),
      _mask(nullptr),
      _trackPerScaleMasks(false),
      _usePerScaleMasks(false) {
  // Make all FFTWF plan calls inside ParallelDeconvolution
  // thread safe.
  schaapcommon::fft::MakeFftwfPlannerThreadSafe();
}

ParallelDeconvolution::~ParallelDeconvolution() {}

ComponentList ParallelDeconvolution::GetComponentList(
    const DeconvolutionTable& table) const {
  // TODO make this work with subimages
  ComponentList list;
  if (_settings.useMultiscale) {
    // If no parallel deconvolution was used, the component list must be
    // retrieved from the deconvolution algorithm.
    if (_algorithms.size() == 1) {
      list = static_cast<MultiScaleAlgorithm*>(_algorithms.front().get())
                 ->GetComponentList();
    } else {
      list = *_componentList;
    }
  } else {
    const size_t w = _settings.trimmedImageWidth;
    const size_t h = _settings.trimmedImageHeight;
    ImageSet modelSet(table, _settings.squaredJoins,
                      _settings.linkedPolarizations, w, h);
    modelSet.LoadAndAverage(false);
    list = ComponentList(w, h, modelSet);
  }
  list.MergeDuplicates();
  return list;
}

const DeconvolutionAlgorithm& ParallelDeconvolution::MaxScaleCountAlgorithm()
    const {
  if (_settings.useMultiscale) {
    MultiScaleAlgorithm* maxAlgorithm =
        static_cast<MultiScaleAlgorithm*>(_algorithms.front().get());
    for (size_t i = 1; i != _algorithms.size(); ++i) {
      MultiScaleAlgorithm* mAlg =
          static_cast<MultiScaleAlgorithm*>(_algorithms[i].get());
      if (mAlg->ScaleCount() > maxAlgorithm->ScaleCount()) {
        maxAlgorithm = mAlg;
      }
    }
    return *maxAlgorithm;
  } else {
    return FirstAlgorithm();
  }
}

void ParallelDeconvolution::SetAlgorithm(
    std::unique_ptr<DeconvolutionAlgorithm> algorithm) {
  if (_settings.parallelDeconvolutionMaxSize == 0) {
    _algorithms.resize(1);
    _algorithms.front() = std::move(algorithm);
  } else {
    const size_t width = _settings.trimmedImageWidth;
    const size_t height = _settings.trimmedImageHeight;
    size_t maxSubImageSize = _settings.parallelDeconvolutionMaxSize;
    _horImages = (width + maxSubImageSize - 1) / maxSubImageSize,
    _verImages = (height + maxSubImageSize - 1) / maxSubImageSize;
    _algorithms.resize(_horImages * _verImages);
    _algorithms.front() = std::move(algorithm);
    size_t threadsPerAlg =
        (_settings.parallelDeconvolutionMaxThreads + _algorithms.size() - 1) /
        _algorithms.size();
    _algorithms.front()->SetThreadCount(threadsPerAlg);
    Logger::Debug << "Parallel deconvolution will use " << _algorithms.size()
                  << " subimages.\n";
    for (size_t i = 1; i != _algorithms.size(); ++i)
      _algorithms[i] = _algorithms.front()->Clone();
  }
}

void ParallelDeconvolution::SetRMSFactorImage(Image&& image) {
  if (_settings.parallelDeconvolutionMaxSize == 0)
    _algorithms.front()->SetRMSFactorImage(std::move(image));
  else
    _rmsImage = std::move(image);
}

void ParallelDeconvolution::SetThreshold(double threshold) {
  for (auto& alg : _algorithms) alg->SetThreshold(threshold);
}

void ParallelDeconvolution::SetAutoMaskMode(bool trackPerScaleMasks,
                                            bool usePerScaleMasks) {
  _trackPerScaleMasks = trackPerScaleMasks;
  _usePerScaleMasks = usePerScaleMasks;
  for (auto& alg : _algorithms) {
    class MultiScaleAlgorithm& algorithm =
        static_cast<class MultiScaleAlgorithm&>(*alg);
    algorithm.SetAutoMaskMode(trackPerScaleMasks, usePerScaleMasks);
  }
}

void ParallelDeconvolution::SetCleanMask(const bool* mask) {
  if (_algorithms.size() == 1)
    _algorithms.front()->SetCleanMask(mask);
  else
    _mask = mask;
}

void ParallelDeconvolution::SetSpectrallyForcedImages(
    std::vector<Image>&& images) {
  if (_algorithms.size() == 1)
    _algorithms.front()->SetSpectrallyForcedImages(std::move(images));
  else
    _spectrallyForcedImages = std::move(images);
}

void ParallelDeconvolution::runSubImage(
    SubImage& subImg, ImageSet& dataImage, const ImageSet& modelImage,
    ImageSet& resultModel, const std::vector<aocommon::Image>& psfImages,
    double majorIterThreshold, bool findPeakOnly, std::mutex& mutex) {
  const size_t width = _settings.trimmedImageWidth;
  const size_t height = _settings.trimmedImageHeight;

  std::unique_ptr<ImageSet> subModel, subData;
  {
    std::lock_guard<std::mutex> lock(mutex);
    subData = dataImage.Trim(subImg.x, subImg.y, subImg.x + subImg.width,
                             subImg.y + subImg.height, width);
    // Because the model of this subimage might extend outside of its boundaries
    // (because of multiscale components), the model is placed back on the image
    // by adding its values. This requires that values outside the boundary are
    // set to zero at this point, otherwise multiple subimages could add the
    // same sources.
    subModel = modelImage.TrimMasked(
        subImg.x, subImg.y, subImg.x + subImg.width, subImg.y + subImg.height,
        width, subImg.boundaryMask.data());
  }

  // Construct the smaller psfs
  std::vector<Image> subPsfs;
  subPsfs.reserve(psfImages.size());
  for (size_t i = 0; i != psfImages.size(); ++i) {
    subPsfs.emplace_back(psfImages[i].Trim(subImg.width, subImg.height));
  }
  _algorithms[subImg.index]->SetCleanMask(subImg.mask.data());

  // Construct smaller RMS image if necessary
  if (!_rmsImage.Empty()) {
    Image subRmsImage =
        _rmsImage.TrimBox(subImg.x, subImg.y, subImg.width, subImg.height);
    _algorithms[subImg.index]->SetRMSFactorImage(std::move(subRmsImage));
  }

  // If a forced spectral image is active, trim it to the subimage size
  if (!_spectrallyForcedImages.empty()) {
    std::vector<Image> subSpectralImages(_spectrallyForcedImages.size());
    for (size_t i = 0; i != _spectrallyForcedImages.size(); ++i) {
      subSpectralImages[i] = _spectrallyForcedImages[i].TrimBox(
          subImg.x, subImg.y, subImg.width, subImg.height);
    }
    _algorithms[subImg.index]->SetSpectrallyForcedImages(
        std::move(subSpectralImages));
  }

  size_t maxNIter = _algorithms[subImg.index]->MaxNIter();
  if (findPeakOnly)
    _algorithms[subImg.index]->SetMaxNIter(0);
  else
    _algorithms[subImg.index]->SetMajorIterThreshold(majorIterThreshold);

  if (_usePerScaleMasks || _trackPerScaleMasks) {
    std::lock_guard<std::mutex> lock(mutex);
    MultiScaleAlgorithm& msAlg =
        static_cast<class MultiScaleAlgorithm&>(*_algorithms[subImg.index]);
    // During the first iteration, msAlg will not have scales/masks yet and the
    // nr scales has also not been determined yet.
    if (!_scaleMasks.empty()) {
      // Here we set the scale mask for the multiscale algorithm.
      // The maximum number of scales in the previous iteration can be found by
      // _scaleMasks.size() Not all msAlgs might have used that many scales, so
      // we have to take this into account
      msAlg.SetScaleMaskCount(
          std::max(msAlg.GetScaleMaskCount(), _scaleMasks.size()));
      for (size_t i = 0; i != msAlg.GetScaleMaskCount(); ++i) {
        aocommon::UVector<bool>& output = msAlg.GetScaleMask(i);
        output.assign(subImg.width * subImg.height, false);
        if (i < _scaleMasks.size())
          Image::TrimBox(output.data(), subImg.x, subImg.y, subImg.width,
                         subImg.height, _scaleMasks[i].data(), width, height);
      }
    }
  }

  subImg.peak = _algorithms[subImg.index]->ExecuteMajorIteration(
      *subData, *subModel, subPsfs, subImg.reachedMajorThreshold);

  // Since this was an RMS image specifically for this subimage size, we free it
  // immediately
  _algorithms[subImg.index]->SetRMSFactorImage(Image());

  if (_trackPerScaleMasks) {
    std::lock_guard<std::mutex> lock(mutex);
    MultiScaleAlgorithm& msAlg =
        static_cast<class MultiScaleAlgorithm&>(*_algorithms[subImg.index]);
    if (_scaleMasks.empty()) {
      _scaleMasks.resize(msAlg.ScaleCount());
      for (aocommon::UVector<bool>& scaleMask : _scaleMasks)
        scaleMask.assign(width * height, false);
    }
    for (size_t i = 0; i != msAlg.ScaleCount(); ++i) {
      const aocommon::UVector<bool>& msMask = msAlg.GetScaleMask(i);
      if (i < _scaleMasks.size())
        Image::CopyMasked(_scaleMasks[i].data(), subImg.x, subImg.y, width,
                          msMask.data(), subImg.width, subImg.height,
                          subImg.boundaryMask.data());
    }
  }

  if (_settings.saveSourceList && _settings.useMultiscale) {
    std::lock_guard<std::mutex> lock(mutex);
    MultiScaleAlgorithm& algorithm =
        static_cast<MultiScaleAlgorithm&>(*_algorithms[subImg.index]);
    if (!_componentList)
      _componentList.reset(new ComponentList(
          width, height, algorithm.ScaleCount(), dataImage.size()));
    _componentList->Add(algorithm.GetComponentList(), subImg.x, subImg.y);
    algorithm.ClearComponentList();
  }

  if (findPeakOnly) {
    _algorithms[subImg.index]->SetMaxNIter(maxNIter);
  } else {
    std::lock_guard<std::mutex> lock(mutex);
    dataImage.CopyMasked(*subData, subImg.x, subImg.y,
                         subImg.boundaryMask.data());
    resultModel.AddSubImage(*subModel, subImg.x, subImg.y);
  }
}

void ParallelDeconvolution::ExecuteMajorIteration(
    ImageSet& dataImage, ImageSet& modelImage,
    const std::vector<aocommon::Image>& psfImages,
    bool& reachedMajorThreshold) {
  if (_algorithms.size() == 1) {
    aocommon::ForwardingLogReceiver fwdReceiver;
    _algorithms.front()->SetLogReceiver(fwdReceiver);
    _algorithms.front()->ExecuteMajorIteration(dataImage, modelImage, psfImages,
                                               reachedMajorThreshold);
  } else {
    executeParallelRun(dataImage, modelImage, psfImages, reachedMajorThreshold);
  }
}

void ParallelDeconvolution::executeParallelRun(
    ImageSet& dataImage, ImageSet& modelImage,
    const std::vector<aocommon::Image>& psfImages,
    bool& reachedMajorThreshold) {
  const size_t width = dataImage.Width();
  const size_t height = dataImage.Height();
  const size_t avgHSubImageSize = width / _horImages;
  const size_t avgVSubImageSize = height / _verImages;

  Image image(width, height);
  Image dividingLine(width, height, 0.0);
  aocommon::UVector<bool> largeScratchMask(width * height);
  dataImage.GetLinearIntegrated(image);

  math::DijkstraSplitter divisor(width, height);

  struct VerticalArea {
    aocommon::UVector<bool> mask;
    size_t x, width;
  };
  std::vector<VerticalArea> verticalAreas(_horImages);

  Logger::Info << "Calculating edge paths...\n";
  aocommon::ParallelFor<size_t> splitLoop(_settings.threadCount);

  // Divide into columns (i.e. construct the vertical lines)
  splitLoop.Run(1, _horImages, [&](size_t divNr, size_t) {
    size_t splitStart = width * divNr / _horImages - avgHSubImageSize / 4,
           splitEnd = width * divNr / _horImages + avgHSubImageSize / 4;
    divisor.DivideVertically(image.Data(), dividingLine.Data(), splitStart,
                             splitEnd);
  });
  for (size_t divNr = 0; divNr != _horImages; ++divNr) {
    size_t midX = divNr * width / _horImages + avgHSubImageSize / 2;
    VerticalArea& area = verticalAreas[divNr];
    divisor.FloodVerticalArea(dividingLine.Data(), midX,
                              largeScratchMask.data(), area.x, area.width);
    area.mask.resize(area.width * height);
    Image::TrimBox(area.mask.data(), area.x, 0, area.width, height,
                   largeScratchMask.data(), width, height);
  }

  // Make the rows (horizontal lines)
  dividingLine = 0.0f;
  splitLoop.Run(1, _verImages, [&](size_t divNr, size_t) {
    size_t splitStart = height * divNr / _verImages - avgVSubImageSize / 4,
           splitEnd = height * divNr / _verImages + avgVSubImageSize / 4;
    divisor.DivideHorizontally(image.Data(), dividingLine.Data(), splitStart,
                               splitEnd);
  });

  Logger::Info << "Calculating bounding boxes and submasks...\n";

  // Find the bounding boxes and clean masks for each subimage
  aocommon::UVector<bool> mask(width * height);
  std::vector<SubImage> subImages;
  for (size_t y = 0; y != _verImages; ++y) {
    size_t midY = y * height / _verImages + avgVSubImageSize / 2;
    size_t hAreaY, hAreaWidth;
    divisor.FloodHorizontalArea(dividingLine.Data(), midY,
                                largeScratchMask.data(), hAreaY, hAreaWidth);

    for (size_t x = 0; x != _horImages; ++x) {
      subImages.emplace_back();
      SubImage& subImage = subImages.back();
      subImage.index = subImages.size() - 1;
      const VerticalArea& vArea = verticalAreas[x];
      divisor.GetBoundingMask(vArea.mask.data(), vArea.x, vArea.width,
                              largeScratchMask.data(), mask.data(), subImage.x,
                              subImage.y, subImage.width, subImage.height);
      Logger::Debug << "Subimage " << subImages.size() << " at (" << subImage.x
                    << "," << subImage.y << ") - ("
                    << subImage.x + subImage.width << ","
                    << subImage.y + subImage.height << ")\n";
      subImage.mask.resize(subImage.width * subImage.height);
      Image::TrimBox(subImage.mask.data(), subImage.x, subImage.y,
                     subImage.width, subImage.height, mask.data(), width,
                     height);
      subImage.boundaryMask = subImage.mask;
      // If a user mask is active, take the union of that mask with the boundary
      // mask (note that 'mask' is reused as a scratch space)
      if (_mask != nullptr) {
        Image::TrimBox(mask.data(), subImage.x, subImage.y, subImage.width,
                       subImage.height, _mask, width, height);
        for (size_t i = 0; i != subImage.mask.size(); ++i)
          subImage.mask[i] = subImage.mask[i] && mask[i];
      }
    }
  }
  verticalAreas.clear();

  // Initialize loggers
  std::mutex mutex;
  _logs.Initialize(_horImages, _verImages);
  for (size_t i = 0; i != _algorithms.size(); ++i)
    _algorithms[i]->SetLogReceiver(_logs[i]);

  // Find the starting peak over all subimages
  aocommon::ParallelFor<size_t> loop(_settings.parallelDeconvolutionMaxThreads);
  ImageSet resultModel(modelImage, modelImage.Width(), modelImage.Height());
  resultModel = 0.0;
  loop.Run(0, _algorithms.size(), [&](size_t index, size_t) {
    _logs.Activate(index);
    runSubImage(subImages[index], dataImage, modelImage, resultModel, psfImages,
                0.0, true, mutex);
    _logs.Deactivate(index);

    _logs[index].Mute(false);
    _logs[index].Info << "Sub-image " << index << " returned peak position.\n";
    _logs[index].Mute(true);
  });
  double maxValue = 0.0;
  size_t indexOfMax = 0;
  for (SubImage& img : subImages) {
    if (img.peak > maxValue) {
      maxValue = img.peak;
      indexOfMax = img.index;
    }
  }
  Logger::Info << "Subimage " << (indexOfMax + 1) << " has maximum peak of "
               << aocommon::units::FluxDensity::ToNiceString(maxValue) << ".\n";
  double mIterThreshold = maxValue * (1.0 - _settings.deconvolutionMGain);

  // Run the deconvolution
  loop.Run(0, _algorithms.size(), [&](size_t index, size_t) {
    _logs.Activate(index);
    runSubImage(subImages[index], dataImage, modelImage, resultModel, psfImages,
                mIterThreshold, false, mutex);
    _logs.Deactivate(index);

    _logs[index].Mute(false);
    _logs[index].Info << "Sub-image " << index
                      << " finished its deconvolution iteration.\n";
    _logs[index].Mute(true);
  });
  modelImage.SetImages(std::move(resultModel));

  _rmsImage.Reset();

  size_t subImagesFinished = 0;
  reachedMajorThreshold = false;
  bool reachedMaxNIter = false;
  for (SubImage& img : subImages) {
    if (!img.reachedMajorThreshold) ++subImagesFinished;
    if (_algorithms[img.index]->IterationNumber() >=
        _algorithms[img.index]->MaxNIter())
      reachedMaxNIter = true;
  }
  Logger::Info << subImagesFinished << " / " << subImages.size()
               << " sub-images finished";
  reachedMajorThreshold = (subImagesFinished != subImages.size());
  if (reachedMajorThreshold && !reachedMaxNIter)
    Logger::Info << ": Continue next major iteration.\n";
  else if (reachedMajorThreshold && reachedMaxNIter) {
    Logger::Info << ", but nr. of iterations reached at least once: "
                    "Deconvolution finished.\n";
    reachedMajorThreshold = false;
  } else
    Logger::Info << ": Deconvolution finished.\n";
}
}  // namespace radler::algorithms