// Copyright (C) 2022 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

#include "component_list.h"

#include "algorithms/deconvolution_algorithm.h"
#include "algorithms/multiscale_algorithm.h"

#include <aocommon/imagecoordinates.h>

#include "radler.h"
#include "utils/write_model.h"

using aocommon::ImageCoordinates;

namespace radler {
void ComponentList::WriteSources(const Radler& radler,
                                 const std::string& filename,
                                 long double pixel_scale_x,
                                 long double pixel_scale_y,
                                 long double phase_centre_ra,
                                 long double phase_centre_dec) const {
  const algorithms::DeconvolutionAlgorithm& deconvolution_algorithm =
      radler.MaxScaleCountAlgorithm();
  if (const auto* multiscale_algorithm =
          dynamic_cast<const algorithms::MultiScaleAlgorithm*>(
              &deconvolution_algorithm)) {
    Write(filename, *multiscale_algorithm, pixel_scale_x, pixel_scale_y,
          phase_centre_ra, phase_centre_dec);
  } else {
    WriteSingleScale(filename, deconvolution_algorithm, pixel_scale_x,
                     pixel_scale_y, phase_centre_ra, phase_centre_dec);
  }
}

void ComponentList::Write(const std::string& filename,
                          const algorithms::MultiScaleAlgorithm& multiscale,
                          long double pixelScaleX, long double pixelScaleY,
                          long double phaseCentreRA,
                          long double phaseCentreDec) const {
  aocommon::UVector<double> scaleSizes(NScales());
  for (size_t scaleIndex = 0; scaleIndex != NScales(); ++scaleIndex)
    scaleSizes[scaleIndex] = multiscale.ScaleSize(scaleIndex);
  write(filename, multiscale.Fitter(), scaleSizes, pixelScaleX, pixelScaleY,
        phaseCentreRA, phaseCentreDec);
}

void ComponentList::WriteSingleScale(
    const std::string& filename,
    const algorithms::DeconvolutionAlgorithm& algorithm,
    long double pixelScaleX, long double pixelScaleY, long double phaseCentreRA,
    long double phaseCentreDec) const {
  aocommon::UVector<double> scaleSizes(1, 0);
  write(filename, algorithm.Fitter(), scaleSizes, pixelScaleX, pixelScaleY,
        phaseCentreRA, phaseCentreDec);
}

void ComponentList::write(const std::string& filename,
                          const schaapcommon::fitters::SpectralFitter& fitter,
                          const aocommon::UVector<double>& scaleSizes,
                          long double pixelScaleX, long double pixelScaleY,
                          long double phaseCentreRA,
                          long double phaseCentreDec) const {
  if (_componentsAddedSinceLastMerge != 0) {
    throw std::runtime_error(
        "ComponentList::write called while there are yet unmerged components. "
        "Run ComponentList::MergeDuplicates() first.");
  }

  if (fitter.Mode() == schaapcommon::fitters::SpectralFittingMode::NoFitting &&
      _nFrequencies > 1)
    throw std::runtime_error(
        "Can't write component list, because you have not specified a spectral "
        "fitting method. You probably want to add '-fit-spectral-pol'.");

  std::ofstream file(filename);
  bool useLogSI = false;
  switch (fitter.Mode()) {
    case schaapcommon::fitters::SpectralFittingMode::NoFitting:
    case schaapcommon::fitters::SpectralFittingMode::Polynomial:
      useLogSI = false;
      break;
    case schaapcommon::fitters::SpectralFittingMode::LogPolynomial:
      useLogSI = true;
      break;
  }
  utils::WriteHeaderForSpectralTerms(file, fitter.ReferenceFrequency());
  std::vector<float> terms;
  for (size_t scaleIndex = 0; scaleIndex != NScales(); ++scaleIndex) {
    const ScaleList& list = _listPerScale[scaleIndex];
    size_t componentIndex = 0;
    const double scale = scaleSizes[scaleIndex];
    // Using the FWHM formula for a Gaussian
    const double
        fwhm =
            2.0L * sqrtl(2.0L * logl(2.0L)) *
            algorithms::multiscale::MultiScaleTransforms::GaussianSigma(scale),
        scaleFWHML = fwhm * pixelScaleX * (180.0 * 60.0 * 60.0 / M_PI),
        scaleFWHMM = fwhm * pixelScaleY * (180.0 * 60.0 * 60.0 / M_PI);
    size_t valueIndex = 0;
    for (size_t index = 0; index != list.positions.size(); ++index) {
      const size_t x = list.positions[index].x;
      const size_t y = list.positions[index].y;
      aocommon::UVector<float> spectrum(_nFrequencies);
      for (size_t frequency = 0; frequency != _nFrequencies; ++frequency) {
        spectrum[frequency] = list.values[valueIndex];
        ++valueIndex;
      }
      if (_nFrequencies == 1)
        terms.assign(1, spectrum[0]);
      else
        fitter.Fit(terms, spectrum.data(), x, y);
      long double l, m;
      ImageCoordinates::XYToLM<long double>(x, y, pixelScaleX, pixelScaleY,
                                            _width, _height, l, m);
      long double ra, dec;
      ImageCoordinates::LMToRaDec(l, m, phaseCentreRA, phaseCentreDec, ra, dec);
      std::ostringstream name;
      name << 's' << scaleIndex << 'c' << componentIndex;
      if (scale == 0.0)
        radler::utils::WritePolynomialPointComponent(
            file, name.str(), ra, dec, useLogSI, terms,
            fitter.ReferenceFrequency());
      else {
        radler::utils::WritePolynomialGaussianComponent(
            file, name.str(), ra, dec, useLogSI, terms,
            fitter.ReferenceFrequency(), scaleFWHML, scaleFWHMM, 0.0);
      }
      ++componentIndex;
    }
  }
}

void ComponentList::loadFromImageSet(ImageSet& imageSet, size_t scaleIndex) {
  _componentsAddedSinceLastMerge = 0;
  _listPerScale[scaleIndex].positions.clear();
  _listPerScale[scaleIndex].values.clear();
  for (size_t y = 0; y != _height; ++y) {
    const size_t rowIndex = y * _width;
    for (size_t x = 0; x != _width; ++x) {
      const size_t posIndex = rowIndex + x;
      bool isNonZero = false;
      for (size_t imageIndex = 0; imageIndex != imageSet.size(); ++imageIndex) {
        if (imageSet[imageIndex][posIndex] != 0.0) {
          isNonZero = true;
          break;
        }
      }
      if (isNonZero) {
        _listPerScale[scaleIndex].positions.emplace_back(x, y);
        for (size_t imageIndex = 0; imageIndex != imageSet.size(); ++imageIndex)
          _listPerScale[scaleIndex].values.push_back(
              imageSet[imageIndex][posIndex]);
      }
    }
  }
}
}  // namespace radler