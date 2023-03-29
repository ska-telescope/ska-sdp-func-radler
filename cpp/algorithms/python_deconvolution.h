// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_ALGORITHMS_PYTHON_DECONVOLUTION_H_
#define RADLER_ALGORITHMS_PYTHON_DECONVOLUTION_H_

#include <aocommon/uvector.h>

#include <schaapcommon/fitters/spectralfitter.h>

#include "image_set.h"
#include "algorithms/deconvolution_algorithm.h"

namespace pybind11 {
// Forward declarations to keep pybind11 symbols internal.
class scoped_interpreter;
class gil_scoped_release;
class function;
}  // namespace pybind11

namespace radler::algorithms {

class PythonDeconvolution final : public DeconvolutionAlgorithm {
 public:
  PythonDeconvolution(const std::string& filename);

  // TODO(AST-912) Make copy/move operations Google Style compliant.
  PythonDeconvolution(const PythonDeconvolution& other);
  PythonDeconvolution(PythonDeconvolution&&) = delete;
  PythonDeconvolution& operator=(const PythonDeconvolution&) = delete;
  PythonDeconvolution& operator=(PythonDeconvolution&&) = delete;

  ~PythonDeconvolution() override;

  float ExecuteMajorIteration(ImageSet& dirty_set, ImageSet& model_set,
                              const std::vector<aocommon::Image>& psfs,
                              bool& reached_major_threshold) final;

  std::unique_ptr<DeconvolutionAlgorithm> Clone() const final {
    return std::make_unique<PythonDeconvolution>(*this);
  }

 private:
  std::string _filename;
  // A Python interpreter can not be restarted, so the interpreter
  // needs to live for the entire run
  std::shared_ptr<pybind11::scoped_interpreter> _guard;
  std::unique_ptr<pybind11::function> _deconvolveFunction;

  // gil_scoped_release needs to be the last of the python members.
  // This ensures that the GIL is obtained by the destructor before
  // the python objects are destructed.
  std::unique_ptr<pybind11::gil_scoped_release> release_;

  void setBuffer(const ImageSet& imageSet, double* pyPtr);
  void setPsf(const std::vector<aocommon::Image>& psfs, double* pyPtr,
              size_t width, size_t height);
  void getBuffer(ImageSet& imageSet, const double* pyPtr);
};
}  // namespace radler::algorithms

#endif  // RADLER_ALGORITHMS_PYTHON_DECONVOLUTION_H_
