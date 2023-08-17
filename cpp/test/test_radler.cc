// SPDX-License-Identifier: LGPL-3.0-only

#include "radler.h"

#include <array>
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

#include <aocommon/fits/fitsreader.h>
#include <aocommon/image.h>
#include <aocommon/logger.h>

#include "settings.h"
#include "test_config.h"

namespace radler {
/**
 * @brief Boost customization point for logging. See:
 * https://www.boost.org/doc/libs/1_64_0/libs/test/doc/html/boost_test/test_output/test_tools_support_for_logging/testing_tool_output_disable.html
 */
std::ostream& boost_test_print_type(std::ostream& stream,
                                    const AlgorithmType& algorithm_type) {
  switch (algorithm_type) {
    case AlgorithmType::kGenericClean:
      stream << "Generic clean";
      break;
    case AlgorithmType::kMultiscale:
      stream << "Multiscale clean";
      break;
    case AlgorithmType::kMoreSane:
      stream << "More sane clean";
      break;
    case AlgorithmType::kIuwt:
      stream << "Iuwt clean";
      break;
    case AlgorithmType::kPython:
      stream << "Python based deconvolver";
      break;
  }
  return stream;
}

namespace {
const std::size_t kWidth = 64;
const std::size_t kHeight = 64;
const double kBeamSize = 0.0;
const double kPixelScale = 1.0 / 60.0 * (M_PI / 180.0);  // 1amin in rad

void FillPsfAndResidual(aocommon::Image& psf_image,
                        aocommon::Image& residual_image, float factor,
                        int shift_x = 0, int shift_y = 0) {
  assert(psf_image.Size() == residual_image.Size());

  // Shift should leave room for at least one index left, right, above and below
  assert(std::abs(shift_x) < (residual_image.Width() / 2 - 1));
  assert(std::abs(shift_y) < (residual_image.Height() / 2 - 1));

  const size_t center_pixel = kHeight / 2 * kWidth + kWidth / 2;
  const size_t shifted_center_pixel =
      (kHeight / 2 + shift_y) * kWidth + (kWidth / 2 + shift_x);

  // Initialize PSF image
  psf_image = 0.0;
  psf_image[center_pixel] = 1.0;
  psf_image[center_pixel - 1] = 0.25;
  psf_image[center_pixel + 1] = 0.5;
  psf_image[center_pixel - kWidth] = 0.4;
  psf_image[center_pixel + kWidth] = 0.6;

  // Initialize residual image
  residual_image = 0.0;
  residual_image[shifted_center_pixel] = 1.0 * factor;
  residual_image[shifted_center_pixel - 1] = 0.25 * factor;
  residual_image[shifted_center_pixel + 1] = 0.5 * factor;
  residual_image[shifted_center_pixel - kWidth] = 0.4 * factor;
  residual_image[shifted_center_pixel + kWidth] = 0.6 * factor;
}
}  // namespace

struct SettingsFixture {
  SettingsFixture() {
    settings.trimmed_image_width = kWidth;
    settings.trimmed_image_height = kHeight;
    settings.pixel_scale.x = kPixelScale;
    settings.pixel_scale.y = kPixelScale;
    settings.minor_iteration_count = 1000;
    settings.absolute_threshold = 1.0e-8;
  }

  Settings settings;
};

std::array<AlgorithmType, 2> kAlgorithmTypes{
    AlgorithmType::kGenericClean, AlgorithmType::kMultiscale,
    /* Fails AlgorithmType::kIuwt */
};

BOOST_AUTO_TEST_SUITE(radler)

BOOST_DATA_TEST_CASE_F(SettingsFixture, centered_source,
                       boost::unit_test::data::make(kAlgorithmTypes),
                       algorithm_type) {
  // The tested function will output log messages. Unit tests shouldn't output
  // to stdout, so prevent the logged output from appearing:
  aocommon::Logger::SetVerbosity(
      aocommon::Logger::VerbosityLevel::kQuietVerbosity);
  settings.algorithm_type = algorithm_type;

  aocommon::Image psf_image(kWidth, kHeight);
  aocommon::Image residual_image(kWidth, kHeight);
  aocommon::Image model_image(kWidth, kHeight, 0.0);

  const float scale_factor = 2.5;
  const size_t center_pixel = kHeight / 2 * kWidth + kWidth / 2;

  FillPsfAndResidual(psf_image, residual_image, scale_factor);

  bool reached_threshold = false;
  const std::size_t iteration_number = 1;
  Radler radler(settings, psf_image, residual_image, model_image, kBeamSize);
  radler.Perform(reached_threshold, iteration_number);

  for (size_t i = 0; i != residual_image.Size(); ++i) {
    BOOST_CHECK_SMALL(residual_image[i], 2.0e-6f);
    if (i == center_pixel) {
      BOOST_CHECK_CLOSE(model_image[i], psf_image[i] * scale_factor, 1.0e-4);
    } else {
      BOOST_CHECK_SMALL(model_image[i], 2.0e-6f);
    }
  }
}

BOOST_DATA_TEST_CASE_F(SettingsFixture, offcentered_source,
                       boost::unit_test::data::make(kAlgorithmTypes),
                       algorithm_type) {
  // The tested function will output log messages. Unit tests shouldn't output
  // to stdout, so prevent the logged output from appearing:
  aocommon::Logger::SetVerbosity(
      aocommon::Logger::VerbosityLevel::kQuietVerbosity);
  settings.algorithm_type = algorithm_type;
  aocommon::Image psf_image(kWidth, kHeight);
  aocommon::Image residual_image(kWidth, kHeight);
  aocommon::Image model_image(kWidth, kHeight, 0.0);

  const float scale_factor = 2.5;
  const int shift_x = 7;
  const int shift_y = -11;
  const size_t center_pixel = kHeight / 2 * kWidth + kWidth / 2;
  const size_t shifted_center_pixel =
      (kHeight / 2 + shift_y) * kWidth + (kWidth / 2 + shift_x);

  FillPsfAndResidual(psf_image, residual_image, scale_factor, shift_x, shift_y);

  bool reached_threshold = false;
  const std::size_t iteration_number = 1;
  Radler radler(settings, psf_image, residual_image, model_image, kBeamSize);
  radler.Perform(reached_threshold, iteration_number);

  for (size_t i = 0; i != residual_image.Size(); ++i) {
    BOOST_CHECK_SMALL(residual_image[i], 2.0e-6f);
    if (i == shifted_center_pixel) {
      BOOST_CHECK_CLOSE(model_image[i], psf_image[center_pixel] * scale_factor,
                        1.0e-4);
    } else {
      BOOST_CHECK_SMALL(model_image[i], 2.0e-6f);
    }
  }
}

BOOST_AUTO_TEST_CASE(diffuse_source) {
  // The tested function will output log messages. Unit tests shouldn't output
  // to stdout, so prevent the logged output from appearing:
  aocommon::Logger::SetVerbosity(
      aocommon::Logger::VerbosityLevel::kQuietVerbosity);
  aocommon::FitsReader imgReader(VELA_DIRTY_IMAGE_PATH);
  aocommon::FitsReader psfReader(VELA_PSF_PATH);

  aocommon::Image psf_image(psfReader.ImageWidth(), psfReader.ImageHeight());
  aocommon::Image residual_image(imgReader.ImageWidth(),
                                 imgReader.ImageHeight());
  aocommon::Image model_image(imgReader.ImageWidth(), imgReader.ImageHeight(),
                              0.0);

  imgReader.Read(residual_image.Data());
  psfReader.Read(psf_image.Data());

  const double max_dirty_image = residual_image.Max();
  const double rms_dirty_image = residual_image.RMS();

  Settings settings;
  settings.algorithm_type = AlgorithmType::kMultiscale;
  settings.absolute_threshold = 1.0e-8;
  settings.major_iteration_count = 30;
  settings.pixel_scale.x = imgReader.PixelSizeX();
  settings.pixel_scale.y = imgReader.PixelSizeY();
  settings.trimmed_image_width = imgReader.ImageWidth();
  settings.trimmed_image_height = imgReader.ImageHeight();
  settings.minor_iteration_count = 300;
  settings.minor_loop_gain = 0.8;
  settings.auto_mask_sigma = 4.0;

  const double beamScale = imgReader.BeamMajorAxisRad();

  Radler radler(settings, psf_image, residual_image, model_image, beamScale);

  bool reached_threshold = false;
  const int major_iteration_count = 0;
  radler.Perform(reached_threshold, major_iteration_count);

  BOOST_CHECK_LE(radler.IterationNumber(), settings.minor_iteration_count);
  BOOST_CHECK_GE(radler.IterationNumber(), 100);

  const double max_residual = residual_image.Max();
  const double rms_residual = residual_image.RMS();

  // Checks that RMS value in residual image went down at least by 25%. A
  // reduction is expected as we remove the peaks from the dirty image.
  BOOST_CHECK_LT(rms_residual, 0.75 * rms_dirty_image);

  // Checks that the components in the dirty image are reduced in the first
  // iteration. We expect the highest peak to be reduced by 90% in this case.
  BOOST_CHECK_LT(max_residual, 0.1 * max_dirty_image);
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace radler
