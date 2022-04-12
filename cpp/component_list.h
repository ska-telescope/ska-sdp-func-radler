// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_COMPONENT_LIST_H_
#define RADLER_COMPONENT_LIST_H_

#include <vector>

#include <aocommon/image.h>
#include <aocommon/uvector.h>

#include <schaapcommon/fitters/spectralfitter.h>

#include "image_set.h"

namespace radler {
class Radler;

namespace algorithms {
class DeconvolutionAlgorithm;
class MultiScaleAlgorithm;
}  // namespace algorithms

class ComponentList {
 public:
  ComponentList()
      : _width(0),
        _height(0),
        _nFrequencies(0),
        _componentsAddedSinceLastMerge(0),
        _maxComponentsBeforeMerge(0),
        _listPerScale() {}

  /**
   * Constructor for single-scale clean
   */
  ComponentList(size_t width, size_t height, ImageSet& imageSet)
      : _width(width),
        _height(height),
        _nFrequencies(imageSet.size()),
        _componentsAddedSinceLastMerge(0),
        _maxComponentsBeforeMerge(100000),
        _listPerScale(1) {
    loadFromImageSet(imageSet, 0);
  }

  /**
   * Constructor for multi-scale clean
   */
  ComponentList(size_t width, size_t height, size_t nScales,
                size_t nFrequencies)
      : _width(width),
        _height(height),
        _nFrequencies(nFrequencies),
        _componentsAddedSinceLastMerge(0),
        _maxComponentsBeforeMerge(100000),
        _listPerScale(nScales) {}

  struct Position {
    Position(size_t _x, size_t _y) : x(_x), y(_y) {}
    size_t x, y;
  };

  void Add(size_t x, size_t y, size_t scaleIndex, const float* values) {
    _listPerScale[scaleIndex].values.push_back(values, values + _nFrequencies);
    _listPerScale[scaleIndex].positions.emplace_back(x, y);
    ++_componentsAddedSinceLastMerge;
    if (_componentsAddedSinceLastMerge >= _maxComponentsBeforeMerge)
      MergeDuplicates();
  }

  void Add(const ComponentList& other, int offsetX, int offsetY) {
    assert(other._nFrequencies == _nFrequencies);
    if (other.NScales() > NScales()) SetNScales(other.NScales());
    for (size_t scale = 0; scale != other.NScales(); ++scale) {
      const ScaleList& list = other._listPerScale[scale];
      for (size_t i = 0; i != list.positions.size(); ++i) {
        Add(list.positions[i].x + offsetX, list.positions[i].y + offsetY, scale,
            &list.values[i * _nFrequencies]);
      }
    }
  }

  void WriteSources(const Radler& radler, const std::string& filename,
                    long double pixel_scale_x, long double pixel_scale_y,
                    long double phase_centre_ra,
                    long double phase_centre_dec) const;

  /**
   * @brief Write component lists over all scales, typically
   * used for writing components of a multiscale clean.
   */
  void Write(const std::string& filename,
             const algorithms::MultiScaleAlgorithm& multiscale,
             long double pixelScaleX, long double pixelScaleY,
             long double phaseCentreRA, long double phaseCentreDec) const;

  void WriteSingleScale(const std::string& filename,
                        const algorithms::DeconvolutionAlgorithm& algorithm,
                        long double pixelScaleX, long double pixelScaleY,
                        long double phaseCentreRA,
                        long double phaseCentreDec) const;

  void MergeDuplicates() {
    if (_componentsAddedSinceLastMerge != 0) {
      for (size_t scaleIndex = 0; scaleIndex != _listPerScale.size();
           ++scaleIndex) {
        mergeDuplicates(scaleIndex);
      }
      _componentsAddedSinceLastMerge = 0;
    }
  }

  void Clear() {
    for (ScaleList& list : _listPerScale) {
      list.positions.clear();
      list.values.clear();
    }
  }

  size_t Width() const { return _width; }
  size_t Height() const { return _height; }

  size_t ComponentCount(size_t scaleIndex) const {
    return _listPerScale[scaleIndex].positions.size();
  }

  void GetComponent(size_t scaleIndex, size_t index, size_t& x, size_t& y,
                    float* values) const {
    assert(scaleIndex < _listPerScale.size());
    assert(index < _listPerScale[scaleIndex].positions.size());
    x = _listPerScale[scaleIndex].positions[index].x;
    y = _listPerScale[scaleIndex].positions[index].y;
    for (size_t f = 0; f != _nFrequencies; ++f)
      values[f] = _listPerScale[scaleIndex].values[index * _nFrequencies + f];
  }

  /**
   * @brief Multiply the components for a given scale index, position index and
   * channel index with corresponding (primary beam) correction factors.
   */
  inline void MultiplyScaleComponent(size_t scaleIndex, size_t positionIndex,
                                     size_t channel, double correctionFactor) {
    assert(scaleIndex < _listPerScale.size());
    assert(positionIndex < _listPerScale[scaleIndex].positions.size());
    assert(channel < _nFrequencies);
    float& value = _listPerScale[scaleIndex]
                       .values[channel + positionIndex * _nFrequencies];
    value *= correctionFactor;
  }

  /**
   * @brief Get vector of positions per scale index.
   */
  const aocommon::UVector<Position>& GetPositions(size_t scaleIndex) const {
    assert(scaleIndex < _listPerScale.size());
    return _listPerScale[scaleIndex].positions;
  }

  size_t NScales() const { return _listPerScale.size(); }

  size_t NFrequencies() const { return _nFrequencies; }

  void SetNScales(size_t nScales) { _listPerScale.resize(nScales); }

 private:
  struct ScaleList {
    /**
     * This list contains nFrequencies values for each
     * component, such that _positions[i] corresponds with the values
     * starting at _values[i * _nFrequencies].
     */
    aocommon::UVector<float> values;
    aocommon::UVector<Position> positions;
  };

  void write(const std::string& filename,
             const schaapcommon::fitters::SpectralFitter& fitter,
             const aocommon::UVector<double>& scaleSizes,
             long double pixelScaleX, long double pixelScaleY,
             long double phaseCentreRA, long double phaseCentreDec) const;

  void loadFromImageSet(ImageSet& imageSet, size_t scaleIndex);

  void mergeDuplicates(size_t scaleIndex) {
    ScaleList& list = _listPerScale[scaleIndex];
    aocommon::UVector<float> newValues;
    aocommon::UVector<Position> newPositions;

    std::vector<aocommon::Image> images(_nFrequencies);
    for (aocommon::Image& image : images)
      image = aocommon::Image(_width, _height, 0.0);
    size_t valueIndex = 0;
    for (size_t index = 0; index != list.positions.size(); ++index) {
      size_t position =
          list.positions[index].x + list.positions[index].y * _width;
      for (size_t frequency = 0; frequency != _nFrequencies; ++frequency) {
        images[frequency][position] += list.values[valueIndex];
        valueIndex++;
      }
    }

    list.values.clear();
    list.positions.clear();

    for (size_t imageIndex = 0; imageIndex != images.size(); ++imageIndex) {
      aocommon::Image& image = images[imageIndex];
      size_t posIndex = 0;
      for (size_t y = 0; y != _height; ++y) {
        for (size_t x = 0; x != _width; ++x) {
          if (image[posIndex] != 0.0) {
            for (size_t i = 0; i != images.size(); ++i) {
              newValues.push_back(images[i][posIndex]);
              images[i][posIndex] = 0.0;
            }
            newPositions.emplace_back(x, y);
          }
          ++posIndex;
        }
      }
    }
    std::swap(_listPerScale[scaleIndex].values, newValues);
    std::swap(_listPerScale[scaleIndex].positions, newPositions);
  }

  size_t _width;
  size_t _height;
  size_t _nFrequencies;
  size_t _componentsAddedSinceLastMerge;
  size_t _maxComponentsBeforeMerge;
  std::vector<ScaleList> _listPerScale;
};
}  // namespace radler
#endif
