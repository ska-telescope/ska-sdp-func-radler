#include "component_optimization.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_multifit.h>

#include <aocommon/logger.h>

using aocommon::Image;
using aocommon::Logger;

namespace radler::math {
namespace {
std::vector<std::pair<size_t, size_t>> GetActivePositions(const Image model,
                                                          size_t width,
                                                          size_t height) {
  const float* image_ptr = model.Data();
  std::vector<std::pair<size_t, size_t>> active_positions;
  for (size_t y = 0; y != height; ++y) {
    for (size_t x = 0; x != width; ++x) {
      if (*image_ptr != 0.0) {
        active_positions.emplace_back(x, y);
      }
      ++image_ptr;
    }
  }
  return active_positions;
}

}  // namespace

aocommon::Image LinearComponentSolve(
    const std::vector<std::pair<size_t, size_t>>& components,
    const aocommon::Image& image, const aocommon::Image& psf) {
  const size_t width = image.Width();
  const size_t height = image.Height();

  // y = X c
  //   - y is vector of n_data,
  //     n_data = number of data points (pixels considered in image)
  //     y_i = pixel value i
  //   - x is vector of n_data x n_active,
  //     n_active = number of parameters (active pixels)
  //     x_ij = (pixel value i) * (psf value j)

  const size_t n_active = components.size();

  // In this function, we use n_data = n_active, i.e. we only try to minimize
  // the residual of active pixels. Two different approaches are to minimize the
  // RMS over all pixels (so n_data = width x height), or to minimize the RMS
  // over some subset of pixels that are "close" to the active components.
  const size_t n_data = n_active;

  gsl_vector* y = gsl_vector_alloc(n_data);
  for (size_t i = 0; i != n_data; ++i) {
    const std::pair<size_t, size_t> position = components[i];
    const size_t index = position.first + position.second * height;
    gsl_vector_set(y, i, image[index]);
  }

  // The PSF is chosen to wrap around. Because of that, we add the width/height
  // here so that the calculation inside the loop is never negative.
  const size_t mid_x = width + (width / 2);
  const size_t mid_y = height + (height / 2);
  gsl_matrix* x_matrix = gsl_matrix_calloc(n_data, n_active);
  for (size_t image_index = 0; image_index != n_data; ++image_index) {
    const size_t image_x = components[image_index].first;
    const size_t image_y = components[image_index].second;
    for (size_t para_index = 0; para_index != n_active; ++para_index) {
      const size_t pos_x = components[para_index].first;
      const size_t pos_y = components[para_index].second;
      const size_t psf_x = (image_x + mid_x - pos_x) % width;
      const size_t psf_y = (image_y + mid_y - pos_y) % height;

      gsl_matrix_set(x_matrix, image_index, para_index,
                     psf[psf_x + psf_y * width]);
    }
  }

  double chi_sq;
  gsl_vector* c = gsl_vector_calloc(n_active);
  gsl_matrix* cov = gsl_matrix_alloc(n_active, n_active);
  gsl_multifit_linear_workspace* work =
      gsl_multifit_linear_alloc(n_data, n_active);
  const int result = gsl_multifit_linear(x_matrix, y, c, cov, &chi_sq, work);

  gsl_multifit_linear_free(work);

  Image model(width, height, 0.0f);
  if (result == GSL_SUCCESS) {
    for (size_t p = 0; p != n_active; ++p) {
      const size_t pos_x = components[p].first;
      const size_t pos_y = components[p].second;
      model[pos_x + pos_y * width] = gsl_vector_get(c, p);
    }
  } else {
    Logger::Warn << "GSL pixel fitter returned an error: "
                 << gsl_strerror(result) << '\n';
  }

  gsl_matrix_free(x_matrix);
  gsl_vector_free(y);
  gsl_vector_free(c);
  gsl_matrix_free(cov);

  return model;
}

void LinearComponentSolve(Image& model, const Image& image, const Image& psf) {
  const size_t width = model.Width();
  const size_t height = model.Height();
  const std::vector<std::pair<size_t, size_t>> active_list =
      GetActivePositions(model, width, height);

  model += LinearComponentSolve(active_list, image, psf);
}

}  // namespace radler::math
