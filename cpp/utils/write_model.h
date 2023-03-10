// SPDX-License-Identifier: LGPL-3.0-only

#ifndef RADLER_UTILS_MODEL_WRITE_MODEL_H_
#define RADLER_UTILS_MODEL_WRITE_MODEL_H_

#include <fstream>
#include <iomanip>
#include <limits>

#include <aocommon/radeccoord.h>
#include <aocommon/uvector.h>

namespace radler::utils {
inline void WriteHeaderForSpectralTerms(std::ostream& stream,
                                        double reference_frequency) {
  constexpr size_t kMaxDoublePrecision =
      std::numeric_limits<double>::digits10 + 1;
  stream.precision(kMaxDoublePrecision);
  stream << "Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, "
            "ReferenceFrequency='"
         << reference_frequency << "', MajorAxis, MinorAxis, Orientation\n";
}

inline void AddSiTerms(std::ostream& stream,
                       const std::vector<float>& si_terms) {
  assert(!si_terms.empty());
  constexpr size_t kMaxFloatPrecision =
      std::numeric_limits<float>::digits10 + 1;
  stream << std::setprecision(kMaxFloatPrecision) << si_terms.front() << ",[";
  if (si_terms.size() >= 2) {
    stream << si_terms[1];
    for (size_t i = 2; i != si_terms.size(); ++i) {
      stream << ',' << si_terms[i];
    }
  }
  stream << ']';
}

inline void WritePolynomialPointComponent(
    std::ostream& stream, const std::string& name, long double ra,
    long double dec, bool use_log_si,
    const std::vector<float>& polarization_terms,
    double reference_frequency_hz) {
  stream << name << ",POINT," << aocommon::RaDecCoord::RAToString(ra, ':')
         << ',' << aocommon::RaDecCoord::DecToString(dec, '.') << ',';
  AddSiTerms(stream, polarization_terms);
  constexpr size_t kMaxDoublePrecision =
      std::numeric_limits<double>::digits10 + 1;
  stream.precision(kMaxDoublePrecision);
  stream << "," << (use_log_si ? "true" : "false") << ","
         << reference_frequency_hz << ",,,\n";
}

inline void WritePolynomialGaussianComponent(
    std::ostream& stream, const std::string& name, long double ra,
    long double dec, bool use_log_si,
    const std::vector<float>& polarization_terms, double reference_frequency_hz,
    double maj, double min, double position_angle) {
  stream << name << ",GAUSSIAN," << aocommon::RaDecCoord::RAToString(ra, ':')
         << ',' << aocommon::RaDecCoord::DecToString(dec, '.') << ',';
  AddSiTerms(stream, polarization_terms);
  constexpr size_t kMaxDoublePrecision =
      std::numeric_limits<double>::digits10 + 1;
  stream.precision(kMaxDoublePrecision);
  stream << "," << (use_log_si ? "true" : "false") << ","
         << reference_frequency_hz << "," << maj << ',' << min << ','
         << position_angle << "\n";
}
}  // namespace radler::utils
#endif
