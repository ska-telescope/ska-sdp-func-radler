// SPDX-License-Identifier: LGPL-3.0-only

#include "math/component_optimization.h"

#include <boost/test/unit_test.hpp>

using aocommon::Image;

namespace radler::math {
namespace {

void TestGradientDescentSimple(bool use_fft_convolution) {
  constexpr size_t kX[5] = {3, 4, 5, 3, 9};
  constexpr size_t kY[5] = {7, 7, 7, 8, 9};
  constexpr float kFittedValues[5] = {3.0, 1.0, -1.0, 9.0, 21.0};
  constexpr float kStartValues[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  constexpr std::pair<float, float> kExpectedRanges[5] = {
      {1.49, 1.51}, {0.49, 0.51}, {-0.51, -0.49}, {4.49, 4.51}, {10.49, 10.51}};

  Image data(10, 10, 0.0f);
  Image model(data.Width(), data.Height(), 0.0f);
  for (size_t i = 0; i != std::size(kFittedValues); ++i) {
    data.Value(kX[i], kY[i]) = kFittedValues[i];
    model.Value(kX[i], kY[i]) = kStartValues[i];
  }

  Image psf(data.Width(), data.Height(), 0.0f);
  psf.Value(data.Width() / 2, data.Height() / 2) = 2.0;

  GradientDescent(model, data, psf, data.Width() * 2, data.Height() * 2,
                  use_fft_convolution);
  for (size_t y = 0; y != data.Height(); ++y) {
    for (size_t x = 0; x != data.Width(); ++x) {
      bool found = false;
      for (size_t i = 0; i != std::size(kFittedValues); ++i) {
        if (x == kX[i] && y == kY[i]) {
          BOOST_CHECK_GT(model.Value(x, y),
                         kExpectedRanges[i].first + kStartValues[i]);
          BOOST_CHECK_LT(model.Value(x, y),
                         kExpectedRanges[i].second + kStartValues[i]);
          found = true;
        }
      }
      if (!found) BOOST_CHECK_LT(std::fabs(model.Value(x, y)), 1e-6);
    }
  }
}

void TestGradientDescentComplex(bool use_fft_convolution) {
  constexpr size_t kX[5] = {3, 4, 5, 3, 9};
  constexpr size_t kY[5] = {7, 7, 7, 8, 9};
  constexpr float kFittedValues[5] = {3.0, 1.0, -1.0, 9.0, 21.0};
  constexpr float kStartValues[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  // v * psf_a = 3              ; v = 1.5
  // w * psf_b + v * psf_a = 1  ; w = (1 - 1.5 * 0.5) / 2 = 0.125
  // x * psf_b + w * psf_a = -1 ; x = (-1 - 0.125 * 0.5) / 2 = -0.53125
  // y * psf_a = 9              ; y = 9 / 2 = 4.5
  // z * psf_a = 21             ; z = 21 / 2 = 10.5
  constexpr std::pair<float, float> kExpectedRanges[5] = {
      {1.4, 1.65}, {0.0, 1.0}, {-1.0, 0.0}, {4.0, 4.7}, {10.3, 10.7}};

  Image data(10, 10, 0.0f);
  Image model(data.Width(), data.Height(), 0.0f);
  for (size_t i = 0; i != std::size(kFittedValues); ++i) {
    data.Value(kX[i], kY[i]) = kFittedValues[i];
    model.Value(kX[i], kY[i]) = kStartValues[i];
  }

  Image psf(data.Width(), data.Height(), 0.0f);
  psf.Value(data.Width() / 2, data.Height() / 2) = 2.0;
  psf.Value(data.Width() / 2 + 1, data.Height() / 2) = 0.5;

  GradientDescent(model, data, psf, data.Width() * 2, data.Height() * 2,
                  use_fft_convolution);
  for (size_t y = 0; y != data.Height(); ++y) {
    for (size_t x = 0; x != data.Width(); ++x) {
      bool found = false;
      for (size_t i = 0; i != std::size(kFittedValues); ++i) {
        if (x == kX[i] && y == kY[i]) {
          BOOST_CHECK_GT(model.Value(x, y),
                         kExpectedRanges[i].first + kStartValues[i]);
          BOOST_CHECK_LT(model.Value(x, y),
                         kExpectedRanges[i].second + kStartValues[i]);
          found = true;
        }
      }
      if (!found) BOOST_CHECK_LT(std::fabs(model.Value(x, y)), 1e-6);
    }
  }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(component_optimization)

BOOST_AUTO_TEST_CASE(single_fit_simple) {
  const float kStartValue = 1.0;
  const float kFittedValue = 3.0;

  // Simple case: one active pixel that needs to be fitted, with a simple PSF.
  Image data(10, 10, 0.0f);
  data.Value(3, 7) = kFittedValue;

  Image model(data.Width(), data.Height(), 0.0f);
  model.Value(3, 7) = kStartValue;

  Image psf(data.Width(), data.Height(), 0.0f);
  psf.Value(5, 5) = 1.0;

  LinearComponentSolve(model, data, psf);
  for (size_t y = 0; y != data.Height(); ++y) {
    for (size_t x = 0; x != data.Width(); ++x) {
      if (x == 3 && y == 7)
        BOOST_CHECK_CLOSE_FRACTION(model.Value(x, y),
                                   kStartValue + kFittedValue, 1e-6);
      else
        BOOST_CHECK_LT(std::fabs(model.Value(x, y)), 1e-6);
    }
  }
}

BOOST_AUTO_TEST_CASE(multi_fit_simple) {
  const size_t kX[4] = {3, 4, 3, 9};
  const size_t kY[4] = {7, 7, 8, 9};
  const float kFittedValues[4] = {3.0, 0.0, 9.0, 21.0};
  const float kStartValues[4] = {1.0, 2.0, 3.0, 4.0};

  Image data(10, 10, 0.0f);
  Image model(data.Width(), data.Height(), 0.0f);
  for (size_t i = 0; i != std::size(kFittedValues); ++i) {
    data.Value(kX[i], kY[i]) = kFittedValues[i];
    model.Value(kX[i], kY[i]) = kStartValues[i];
  }

  Image psf(data.Width(), data.Height(), 0.0f);
  psf.Value(5, 5) = 1.0;

  LinearComponentSolve(model, data, psf);
  for (size_t y = 0; y != data.Height(); ++y) {
    for (size_t x = 0; x != data.Width(); ++x) {
      bool found = false;
      for (size_t i = 0; i != std::size(kFittedValues); ++i) {
        if (x == kX[i] && y == kY[i]) {
          BOOST_CHECK_CLOSE_FRACTION(model.Value(x, y),
                                     kStartValues[i] + kFittedValues[i], 1e-6);
          found = true;
        }
      }
      if (!found) BOOST_CHECK_LT(std::fabs(model.Value(x, y)), 1e-6);
    }
  }
}

BOOST_AUTO_TEST_CASE(multi_fit_with_overlap) {
  constexpr size_t kX[5] = {3, 4, 5, 3, 9};
  constexpr size_t kY[5] = {7, 7, 7, 8, 9};
  constexpr float kFittedValues[5] = {3.0, 1.0, -1.0, 9.0, 21.0};
  constexpr float kStartValues[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  // v * psf_a = 3              ; v = 1.5
  // w * psf_b + v * psf_a = 1  ; w = (1 - 1.5 * 0.5) / 2 = 0.125
  // x * psf_b + w * psf_a = -1 ; x = (-1 - 0.125 * 0.5) / 2 = -0.53125
  // y * psf_a = 9              ; y = 9 / 2 = 4.5
  // z * psf_a = 21             ; z = 21 / 2 = 10.5
  constexpr float kExpectedValues[5] = {1.5, 0.125, -0.53125, 4.5, 10.5};

  Image data(10, 10, 0.0f);
  Image model(data.Width(), data.Height(), 0.0f);
  for (size_t i = 0; i != std::size(kFittedValues); ++i) {
    data.Value(kX[i], kY[i]) = kFittedValues[i];
    model.Value(kX[i], kY[i]) = kStartValues[i];
  }

  Image psf(data.Width(), data.Height(), 0.0f);
  psf.Value(5, 5) = 2.0;  // This is psf_a in the equations above
  psf.Value(6, 5) = 0.5;  // psf_b

  LinearComponentSolve(model, data, psf);
  for (size_t y = 0; y != data.Height(); ++y) {
    for (size_t x = 0; x != data.Width(); ++x) {
      bool found = false;
      for (size_t i = 0; i != std::size(kFittedValues); ++i) {
        if (x == kX[i] && y == kY[i]) {
          BOOST_CHECK_CLOSE_FRACTION(
              model.Value(x, y), kExpectedValues[i] + kStartValues[i], 1e-6);
          found = true;
        }
      }
      if (!found) BOOST_CHECK_LT(std::fabs(model.Value(x, y)), 1e-6);
    }
  }
}

BOOST_AUTO_TEST_CASE(gradient_descent_simple_psf) {
  TestGradientDescentSimple(false);
}

BOOST_AUTO_TEST_CASE(gradient_descent_simple_psf_with_fft) {
  TestGradientDescentSimple(true);
}

BOOST_AUTO_TEST_CASE(gradient_descent_complex) {
  TestGradientDescentComplex(false);
}

BOOST_AUTO_TEST_CASE(gradient_descent_complex_with_fft) {
  TestGradientDescentComplex(true);
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace radler::math
