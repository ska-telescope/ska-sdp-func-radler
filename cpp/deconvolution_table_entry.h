// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_DECONVOLUTION_TABLE_ENTRY_H_
#define RADLER_DECONVOLUTION_TABLE_ENTRY_H_

#include <memory>
#include <vector>

#include <aocommon/imageaccessor.h>
#include <aocommon/polarization.h>

namespace radler {
struct DeconvolutionTableEntry {
  double CentralFrequency() const {
    return 0.5 * (band_start_frequency + band_end_frequency);
  }

  /**
   * Index of the entry in its DeconvolutionTable.
   */
  size_t index = 0;

  /**
   * Note that mses might have overlapping frequencies.
   */
  double band_start_frequency = 0.0;
  double band_end_frequency = 0.0;

  aocommon::PolarizationEnum polarization = aocommon::PolarizationEnum::StokesI;

  /**
   * Entries with equal original channel indices are 'joinedly' deconvolved by
   * adding their squared flux density values together. Normally, all the
   * polarizations from a single channel / timestep form such a group.
   *
   * When the number of deconvolution channels is less than the number of
   * original channels, entries in multiple groups are 'joinedly' deconvolved.
   */
  size_t original_channel_index = 0;
  size_t original_interval_index = 0;

  /**
   * A number that scales with the estimated inverse-variance of the image. It
   * can be used when averaging images or fitting functions through the images
   * to get the optimal sensitivity. It is set after the first inversion.
   */
  double image_weight = 0.0;

  /**
   * Image accessor for the PSF image for this entry. This accessor is only used
   * for the first entry of each channel group.
   */
  std::unique_ptr<aocommon::ImageAccessor> psf_accessor;

  /**
   * Image accessor for the model image for this entry.
   */
  std::unique_ptr<aocommon::ImageAccessor> model_accessor;

  /**
   * Image accessor for the residual image for this entry.
   */
  std::unique_ptr<aocommon::ImageAccessor> residual_accessor;
};
}  // namespace radler
#endif
