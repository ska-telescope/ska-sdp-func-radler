// Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RADLER_DECONVOLUTION_SETTINGS_H_
#define RADLER_DECONVOLUTION_SETTINGS_H_

#include <set>

#include <aocommon/polarization.h>
#include <aocommon/system.h>

#include <schaapcommon/fitters/spectralfitter.h>

#include "algorithms/multiscale/multiscale_transforms.h"

namespace radler {
/**
 * @brief The value of LocalRmsMethod describes how the RMS map
 * should be used.
 */
enum class LocalRmsMethod { kNone, kRmsWindow, kRmsAndMinimumWindow };

struct DeconvolutionSettings {
  DeconvolutionSettings();

  /**
   * @{
   * Settings that are duplicates from top level settings, and also used outside
   * deconvolution.
   */
  size_t trimmedImageWidth;
  size_t trimmedImageHeight;
  size_t channelsOut;
  double pixelScaleX;
  double pixelScaleY;
  size_t threadCount;
  std::string prefixName;
  /** @} */

  /**
   * @{
   * These settings strictly pertain to deconvolution only.
   */
  std::set<aocommon::PolarizationEnum> linkedPolarizations;
  size_t parallelDeconvolutionMaxSize;
  size_t parallelDeconvolutionMaxThreads;
  double deconvolutionThreshold;
  double deconvolutionGain;
  double deconvolutionMGain;
  bool autoDeconvolutionThreshold;
  bool autoMask;
  double autoDeconvolutionThresholdSigma;
  double autoMaskSigma;
  LocalRmsMethod localRMSMethod;
  double localRMSWindow;
  std::string localRMSImage;
  bool saveSourceList;
  size_t deconvolutionIterationCount;
  size_t majorIterationCount;
  bool allowNegativeComponents;
  bool stopOnNegativeComponents;
  bool useMultiscale;
  bool useSubMinorOptimization;
  bool squaredJoins;
  double spectralCorrectionFrequency;
  std::vector<float> spectralCorrection;
  bool multiscaleFastSubMinorLoop;
  double multiscaleGain;
  double multiscaleDeconvolutionScaleBias;
  size_t multiscaleMaxScales;
  double multiscaleConvolutionPadding;
  std::vector<double> multiscaleScaleList;
  algorithms::multiscale::Shape multiscaleShapeFunction;
  double deconvolutionBorderRatio;
  std::string fitsDeconvolutionMask;
  std::string casaDeconvolutionMask;
  bool horizonMask;
  double horizonMaskDistance;
  std::string pythonDeconvolutionFilename;
  bool useMoreSaneDeconvolution;
  bool useIUWTDeconvolution;
  bool iuwtSNRTest;
  std::string moreSaneLocation;
  std::string moreSaneArgs;
  std::vector<double> moreSaneSigmaLevels;
  schaapcommon::fitters::SpectralFittingMode spectralFittingMode;
  size_t spectralFittingTerms;
  std::string forcedSpectrumFilename;
  /**
   * The number of channels used during deconvolution. This can be used to
   * image with more channels than deconvolution. Before deconvolution,
   * channels are averaged, and after deconvolution they are interpolated.
   * If it is 0, all channels should be used.
   */
  size_t deconvolutionChannelCount;
  /** @} */
};

inline DeconvolutionSettings::DeconvolutionSettings()
    : trimmedImageWidth(0),
      trimmedImageHeight(0),
      channelsOut(1),
      pixelScaleX(0.0),
      pixelScaleY(0.0),
      threadCount(aocommon::system::ProcessorCount()),
      prefixName("wsclean"),
      linkedPolarizations(),
      parallelDeconvolutionMaxSize(0),
      parallelDeconvolutionMaxThreads(0),
      deconvolutionThreshold(0.0),
      deconvolutionGain(0.1),
      deconvolutionMGain(1.0),
      autoDeconvolutionThreshold(false),
      autoMask(false),
      autoDeconvolutionThresholdSigma(0.0),
      autoMaskSigma(0.0),
      localRMSMethod(LocalRmsMethod::kNone),
      localRMSWindow(25.0),
      localRMSImage(),
      saveSourceList(false),
      deconvolutionIterationCount(0),
      majorIterationCount(20),
      allowNegativeComponents(true),
      stopOnNegativeComponents(false),
      useMultiscale(false),
      useSubMinorOptimization(true),
      squaredJoins(false),
      spectralCorrectionFrequency(0.0),
      spectralCorrection(),
      multiscaleFastSubMinorLoop(true),
      multiscaleGain(0.2),
      multiscaleDeconvolutionScaleBias(0.6),
      multiscaleMaxScales(0),
      multiscaleConvolutionPadding(1.1),
      multiscaleScaleList(),
      multiscaleShapeFunction(
          algorithms::multiscale::Shape::TaperedQuadraticShape),
      deconvolutionBorderRatio(0.0),
      fitsDeconvolutionMask(),
      casaDeconvolutionMask(),
      horizonMask(false),
      horizonMaskDistance(0.0),
      pythonDeconvolutionFilename(),
      useMoreSaneDeconvolution(false),
      useIUWTDeconvolution(false),
      iuwtSNRTest(false),
      moreSaneLocation(),
      moreSaneArgs(),
      spectralFittingMode(
          schaapcommon::fitters::SpectralFittingMode::NoFitting),
      spectralFittingTerms(0),
      forcedSpectrumFilename(),
      deconvolutionChannelCount(0) {}
}  // namespace radler
#endif
