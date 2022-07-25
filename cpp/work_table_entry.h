// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_WORK_TABLE_ENTRY_H_
#define RADLER_WORK_TABLE_ENTRY_H_

#include "psf.h"

#include <iomanip>
#include <memory>
#include <ostream>
#include <vector>

#include <aocommon/imageaccessor.h>
#include <aocommon/polarization.h>

namespace radler {

struct WorkTableEntry {
  double CentralFrequency() const {
    return 0.5 * (band_start_frequency + band_end_frequency);
  }

  /**
   * Index of the entry in its WorkTable.
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
   * Image accessors for the PSF image for this entry.
   *
   * These accessors are only used for the first entry of each channel group,
   * i.e., all polarizations of the same channel share the same PSF.
   */
  std::vector<Psf> psfs{};

  /**
   * Image accessor for the model image for this entry.
   */
  std::unique_ptr<aocommon::ImageAccessor> model_accessor;

  /**
   * Image accessor for the residual image for this entry.
   */
  std::unique_ptr<aocommon::ImageAccessor> residual_accessor;

  friend std::ostream& operator<<(std::ostream& output,
                                  const WorkTableEntry& entry) {
    return output << "  " << std::setw(2) << entry.index << " " << std::setw(3)
                  << aocommon::Polarization::TypeToShortString(
                         entry.polarization)
                  << " " << std::setw(2) << entry.original_channel_index << " "
                  << std::setw(8) << entry.original_interval_index << " "
                  << std::setw(6) << entry.image_weight << " "
                  << round(entry.band_start_frequency * 1e-6) << "-"
                  << round(entry.band_end_frequency * 1e-6) << '\n';
  }
};
}  // namespace radler
#endif
