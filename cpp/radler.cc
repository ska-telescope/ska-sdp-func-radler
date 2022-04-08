// Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "radler.h"

#include <aocommon/fits/fitsreader.h>
#include <aocommon/image.h>
#include <aocommon/imagecoordinates.h>
#include <aocommon/logger.h>
#include <aocommon/units/fluxdensity.h>

#include <schaapcommon/fft/convolution.h>

#include "algorithms/generic_clean.h"
#include "algorithms/iuwt_deconvolution.h"
#include "algorithms/more_sane.h"
#include "algorithms/multiscale_algorithm.h"
#include "algorithms/parallel_deconvolution.h"
#include "algorithms/python_deconvolution.h"
#include "algorithms/simple_clean.h"

#include "image_set.h"
#include "math/rms_image.h"
#include "utils/casa_mask_reader.h"

using aocommon::FitsReader;
using aocommon::FitsWriter;
using aocommon::Image;
using aocommon::ImageCoordinates;
using aocommon::Logger;
using aocommon::units::FluxDensity;

namespace radler {

Radler::Radler(const DeconvolutionSettings& deconvolutionSettings)
    : _settings(deconvolutionSettings),
      _table(),
      _parallelDeconvolution(
          std::make_unique<algorithms::ParallelDeconvolution>(_settings)),
      _autoMaskIsFinished(false),
      _imgWidth(_settings.trimmedImageWidth),
      _imgHeight(_settings.trimmedImageHeight),
      _pixelScaleX(_settings.pixelScaleX),
      _pixelScaleY(_settings.pixelScaleY),
      _autoMask(),
      _beamSize(0.0) {
  // Ensure that all FFTWF plan calls inside Radler are
  // thread safe.
  schaapcommon::fft::MakeFftwfPlannerThreadSafe();
}

Radler::~Radler() { FreeDeconvolutionAlgorithms(); }

ComponentList Radler::GetComponentList() const {
  return _parallelDeconvolution->GetComponentList(*_table);
}

const algorithms::DeconvolutionAlgorithm& Radler::MaxScaleCountAlgorithm()
    const {
  return _parallelDeconvolution->MaxScaleCountAlgorithm();
}

void Radler::Perform(bool& reachedMajorThreshold, size_t majorIterationNr) {
  assert(_table);

  Logger::Info.Flush();
  Logger::Info << " == Deconvolving (" << majorIterationNr << ") ==\n";

  ImageSet residualSet(*_table, _settings.squaredJoins,
                       _settings.linkedPolarizations, _imgWidth, _imgHeight);
  ImageSet modelSet(*_table, _settings.squaredJoins,
                    _settings.linkedPolarizations, _imgWidth, _imgHeight);

  Logger::Debug << "Loading residual images...\n";
  residualSet.LoadAndAverage(true);
  Logger::Debug << "Loading model images...\n";
  modelSet.LoadAndAverage(false);

  Image integrated(_imgWidth, _imgHeight);
  residualSet.GetLinearIntegrated(integrated);
  Logger::Debug << "Calculating standard deviation...\n";
  double stddev = integrated.StdDevFromMAD();
  Logger::Info << "Estimated standard deviation of background noise: "
               << FluxDensity::ToNiceString(stddev) << '\n';
  if (_settings.autoMask && _autoMaskIsFinished) {
    // When we are in the second phase of automasking, don't use
    // the RMS background anymore
    _parallelDeconvolution->SetRMSFactorImage(Image());
  } else {
    if (!_settings.localRMSImage.empty()) {
      Image rmsImage(_imgWidth, _imgHeight);
      FitsReader reader(_settings.localRMSImage);
      reader.Read(rmsImage.Data());
      // Normalize the RMS image
      stddev = rmsImage.Min();
      Logger::Info << "Lowest RMS in image: "
                   << FluxDensity::ToNiceString(stddev) << '\n';
      if (stddev <= 0.0)
        throw std::runtime_error(
            "RMS image can only contain values > 0, but contains values <= "
            "0.0");
      for (float& value : rmsImage) {
        if (value != 0.0) value = stddev / value;
      }
      _parallelDeconvolution->SetRMSFactorImage(std::move(rmsImage));
    } else if (_settings.localRMSMethod != LocalRmsMethod::kNone) {
      Logger::Debug << "Constructing local RMS image...\n";
      Image rmsImage;
      // TODO this should use full beam parameters
      switch (_settings.localRMSMethod) {
        case LocalRmsMethod::kNone:
          assert(false);
          break;
        case LocalRmsMethod::kRmsWindow:
          math::RMSImage::Make(rmsImage, integrated, _settings.localRMSWindow,
                               _beamSize, _beamSize, 0.0, _pixelScaleX,
                               _pixelScaleY, _settings.threadCount);
          break;
        case LocalRmsMethod::kRmsAndMinimumWindow:
          math::RMSImage::MakeWithNegativityLimit(
              rmsImage, integrated, _settings.localRMSWindow, _beamSize,
              _beamSize, 0.0, _pixelScaleX, _pixelScaleY,
              _settings.threadCount);
          break;
      }
      // Normalize the RMS image relative to the threshold so that Jy remains
      // Jy.
      stddev = rmsImage.Min();
      Logger::Info << "Lowest RMS in image: "
                   << FluxDensity::ToNiceString(stddev) << '\n';
      for (float& value : rmsImage) {
        if (value != 0.0) value = stddev / value;
      }
      _parallelDeconvolution->SetRMSFactorImage(std::move(rmsImage));
    }
  }
  if (_settings.autoMask && !_autoMaskIsFinished)
    _parallelDeconvolution->SetThreshold(std::max(
        stddev * _settings.autoMaskSigma, _settings.deconvolutionThreshold));
  else if (_settings.autoDeconvolutionThreshold)
    _parallelDeconvolution->SetThreshold(
        std::max(stddev * _settings.autoDeconvolutionThresholdSigma,
                 _settings.deconvolutionThreshold));
  integrated.Reset();

  Logger::Debug << "Loading PSFs...\n";
  const std::vector<aocommon::Image> psfImages =
      residualSet.LoadAndAveragePSFs();

  if (_settings.useMultiscale) {
    if (_settings.autoMask) {
      if (_autoMaskIsFinished)
        _parallelDeconvolution->SetAutoMaskMode(false, true);
      else
        _parallelDeconvolution->SetAutoMaskMode(true, false);
    }
  } else {
    if (_settings.autoMask && _autoMaskIsFinished) {
      if (_autoMask.empty()) {
        _autoMask.resize(_imgWidth * _imgHeight);
        for (size_t imgIndex = 0; imgIndex != modelSet.size(); ++imgIndex) {
          const aocommon::Image& image = modelSet[imgIndex];
          for (size_t i = 0; i != _imgWidth * _imgHeight; ++i) {
            _autoMask[i] = (image[i] == 0.0) ? false : true;
          }
        }
      }
      _parallelDeconvolution->SetCleanMask(_autoMask.data());
    }
  }

  _parallelDeconvolution->ExecuteMajorIteration(
      residualSet, modelSet, psfImages, reachedMajorThreshold);

  if (!reachedMajorThreshold && _settings.autoMask && !_autoMaskIsFinished) {
    Logger::Info << "Auto-masking threshold reached; continuing next major "
                    "iteration with deeper threshold and mask.\n";
    _autoMaskIsFinished = true;
    reachedMajorThreshold = true;
  }

  if (_settings.majorIterationCount != 0 &&
      majorIterationNr >= _settings.majorIterationCount) {
    reachedMajorThreshold = false;
    Logger::Info << "Maximum number of major iterations was reached: not "
                    "continuing deconvolution.\n";
  }

  if (_settings.deconvolutionIterationCount != 0 &&
      _parallelDeconvolution->FirstAlgorithm().IterationNumber() >=
          _settings.deconvolutionIterationCount) {
    reachedMajorThreshold = false;
    Logger::Info
        << "Maximum number of minor deconvolution iterations was reached: not "
           "continuing deconvolution.\n";
  }

  residualSet.AssignAndStoreResidual();
  modelSet.InterpolateAndStoreModel(
      _parallelDeconvolution->FirstAlgorithm().Fitter(), _settings.threadCount);
}

void Radler::InitializeDeconvolutionAlgorithm(
    std::unique_ptr<DeconvolutionTable> table, double beamSize,
    size_t threadCount) {
  _beamSize = beamSize;
  _autoMaskIsFinished = false;
  _autoMask.clear();
  FreeDeconvolutionAlgorithms();
  _table = std::move(table);
  if (_table->OriginalGroups().empty())
    throw std::runtime_error("Nothing to clean");

  if (!std::isfinite(_beamSize)) {
    Logger::Warn << "No proper beam size available in deconvolution!\n";
    _beamSize = 0.0;
  }

  std::unique_ptr<class algorithms::DeconvolutionAlgorithm> algorithm;

  if (!_settings.pythonDeconvolutionFilename.empty()) {
    algorithm.reset(new algorithms::PythonDeconvolution(
        _settings.pythonDeconvolutionFilename));
  } else if (_settings.useMoreSaneDeconvolution) {
    algorithm.reset(new algorithms::MoreSane(
        _settings.moreSaneLocation, _settings.moreSaneArgs,
        _settings.moreSaneSigmaLevels, _settings.prefixName));
  } else if (_settings.useIUWTDeconvolution) {
    algorithms::IUWTDeconvolution* method = new algorithms::IUWTDeconvolution;
    method->SetUseSNRTest(_settings.iuwtSNRTest);
    algorithm.reset(method);
  } else if (_settings.useMultiscale) {
    algorithms::MultiScaleAlgorithm* msAlgorithm =
        new algorithms::MultiScaleAlgorithm(beamSize, _pixelScaleX,
                                            _pixelScaleY);
    msAlgorithm->SetManualScaleList(_settings.multiscaleScaleList);
    msAlgorithm->SetMultiscaleScaleBias(
        _settings.multiscaleDeconvolutionScaleBias);
    msAlgorithm->SetMaxScales(_settings.multiscaleMaxScales);
    msAlgorithm->SetMultiscaleGain(_settings.multiscaleGain);
    msAlgorithm->SetShape(_settings.multiscaleShapeFunction);
    msAlgorithm->SetTrackComponents(_settings.saveSourceList);
    msAlgorithm->SetConvolutionPadding(_settings.multiscaleConvolutionPadding);
    msAlgorithm->SetUseFastSubMinorLoop(_settings.multiscaleFastSubMinorLoop);
    algorithm.reset(msAlgorithm);
  } else {
    algorithm.reset(
        new algorithms::GenericClean(_settings.useSubMinorOptimization));
  }

  algorithm->SetMaxNIter(_settings.deconvolutionIterationCount);
  algorithm->SetThreshold(_settings.deconvolutionThreshold);
  algorithm->SetGain(_settings.deconvolutionGain);
  algorithm->SetMGain(_settings.deconvolutionMGain);
  algorithm->SetCleanBorderRatio(_settings.deconvolutionBorderRatio);
  algorithm->SetAllowNegativeComponents(_settings.allowNegativeComponents);
  algorithm->SetStopOnNegativeComponents(_settings.stopOnNegativeComponents);
  algorithm->SetThreadCount(threadCount);
  algorithm->SetSpectralFittingMode(_settings.spectralFittingMode,
                                    _settings.spectralFittingTerms);

  ImageSet::CalculateDeconvolutionFrequencies(*_table, _channelFrequencies,
                                              _channelWeights);
  algorithm->InitializeFrequencies(_channelFrequencies, _channelWeights);
  _parallelDeconvolution->SetAlgorithm(std::move(algorithm));

  if (!_settings.forcedSpectrumFilename.empty()) {
    Logger::Debug << "Reading " << _settings.forcedSpectrumFilename << ".\n";
    FitsReader reader(_settings.forcedSpectrumFilename);
    if (reader.ImageWidth() != _imgWidth || reader.ImageHeight() != _imgHeight)
      throw std::runtime_error(
          "The image width of the forced spectrum fits file does not match the "
          "imaging size");
    std::vector<Image> terms(1);
    terms[0] = Image(_imgWidth, _imgHeight);
    reader.Read(terms[0].Data());
    _parallelDeconvolution->SetSpectrallyForcedImages(std::move(terms));
  }

  readMask(*_table);
}

void Radler::FreeDeconvolutionAlgorithms() {
  _parallelDeconvolution->FreeDeconvolutionAlgorithms();
  _table.reset();
}

bool Radler::IsInitialized() const {
  return _parallelDeconvolution->IsInitialized();
}

size_t Radler::IterationNumber() const {
  return _parallelDeconvolution->FirstAlgorithm().IterationNumber();
}

void Radler::RemoveNaNsInPSF(float* psf, size_t width, size_t height) {
  float* endPtr = psf + width * height;
  while (psf != endPtr) {
    if (!std::isfinite(*psf)) *psf = 0.0;
    ++psf;
  }
}

void Radler::readMask(const DeconvolutionTable& groupTable) {
  bool hasMask = false;
  if (!_settings.fitsDeconvolutionMask.empty()) {
    FitsReader maskReader(_settings.fitsDeconvolutionMask, true, true);
    if (maskReader.ImageWidth() != _imgWidth ||
        maskReader.ImageHeight() != _imgHeight)
      throw std::runtime_error(
          "Specified Fits file mask did not have same dimensions as output "
          "image!");
    aocommon::UVector<float> maskData(_imgWidth * _imgHeight);
    if (maskReader.NFrequencies() == 1) {
      Logger::Debug << "Reading mask '" << _settings.fitsDeconvolutionMask
                    << "'...\n";
      maskReader.Read(maskData.data());
    } else if (maskReader.NFrequencies() == _settings.channelsOut) {
      Logger::Debug << "Reading mask '" << _settings.fitsDeconvolutionMask
                    << "' (" << (groupTable.Front().original_channel_index + 1)
                    << ")...\n";
      maskReader.ReadIndex(maskData.data(),
                           groupTable.Front().original_channel_index);
    } else {
      std::stringstream msg;
      msg << "The number of frequencies in the specified fits mask ("
          << maskReader.NFrequencies()
          << ") does not match the number of requested output channels ("
          << _settings.channelsOut << ")";
      throw std::runtime_error(msg.str());
    }
    _cleanMask.assign(_imgWidth * _imgHeight, false);
    for (size_t i = 0; i != _imgWidth * _imgHeight; ++i)
      _cleanMask[i] = (maskData[i] != 0.0);

    hasMask = true;
  } else if (!_settings.casaDeconvolutionMask.empty()) {
    if (_cleanMask.empty()) {
      Logger::Info << "Reading CASA mask '" << _settings.casaDeconvolutionMask
                   << "'...\n";
      _cleanMask.assign(_imgWidth * _imgHeight, false);
      utils::CasaMaskReader maskReader(_settings.casaDeconvolutionMask);
      if (maskReader.Width() != _imgWidth || maskReader.Height() != _imgHeight)
        throw std::runtime_error(
            "Specified CASA mask did not have same dimensions as output "
            "image!");
      maskReader.Read(_cleanMask.data());
    }

    hasMask = true;
  }

  if (_settings.horizonMask) {
    if (!hasMask) {
      _cleanMask.assign(_imgWidth * _imgHeight, true);
      hasMask = true;
    }

    double fovSq = M_PI_2 - _settings.horizonMaskDistance;
    if (fovSq < 0.0) fovSq = 0.0;
    if (fovSq <= M_PI_2)
      fovSq = std::sin(fovSq);
    else  // a negative horizon distance was given
      fovSq = 1.0 - _settings.horizonMaskDistance;
    fovSq = fovSq * fovSq;
    bool* ptr = _cleanMask.data();

    for (size_t y = 0; y != _imgHeight; ++y) {
      for (size_t x = 0; x != _imgWidth; ++x) {
        double l, m;
        ImageCoordinates::XYToLM(x, y, _pixelScaleX, _pixelScaleY, _imgWidth,
                                 _imgHeight, l, m);
        if (l * l + m * m >= fovSq) *ptr = false;
        ++ptr;
      }
    }

    Logger::Info << "Saving horizon mask...\n";
    Image image(_imgWidth, _imgHeight);
    for (size_t i = 0; i != _imgWidth * _imgHeight; ++i)
      image[i] = _cleanMask[i] ? 1.0 : 0.0;

    FitsWriter writer;
    writer.SetImageDimensions(_imgWidth, _imgHeight, _settings.pixelScaleX,
                              _settings.pixelScaleY);
    writer.Write(_settings.prefixName + "-horizon-mask.fits", image.Data());
  }

  if (hasMask) _parallelDeconvolution->SetCleanMask(_cleanMask.data());
}
}  // namespace radler