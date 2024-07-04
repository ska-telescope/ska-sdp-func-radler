# Radler: Radio Astronomical Deconvolution Library

Radler is a library providing functionality for deconvolving astronomical images. The library was split off from WSClean, `https://gitlab.com/aroffringa/wsclean`_ in order to enhance modularity and comparisons.

## Documentation

The Radler-specific documentation can be found here: `https://radler.readthedocs.io/`_, which includes how to interface with Radler from Python or C++. Information about the different methods and when to
use which method (aimed at astronomers) can be found in the WSClean manual, `https://wsclean.readthedocs.io/`_. The WSClean documentation also contains references to scientific papers that describe the methods that are implemented
in Radler.

## Testing
Tests for the core functionality - in particular the different deconvolution algorithms can be found in the `cpp/test` directory. Smaller scale unit tests can be found at namespace level (see, e.g., `cpp/math/test`).

Some example scripts of how the C++ interface can be used, are found in the `cpp/demo` directory.

## License
Radler is released under the LGPL version 3.
