.. _buildinstructions:

Build instructions
==================

Radler can be installed as a stand-alone package, but is also installed as a part of `WSClean <https://wsclean.readthedocs.io>`_. 
If you only want to install WSClean, it is not necessary to build Radler yourself.

Installing from pypi
~~~~~~~~~~~~~~~~~~~~
Radler can be installed from pypi:

::

    pip install radler


Building from source
~~~~~~~~~~~~~~~~~~~~
Radler needs a number of dependencies in order to successfully compile. On a clean (ubuntu 22.04) system,
the dependencies can be installed with (see also the ``docker`` directory):

General packages:

::

    apt update && apt -y install git make cmake libpython3-dev g++ \
    libboost-date-time-dev libhdf5-dev libfftw3-dev libgsl-dev python3-pip

Astronomy-specific packages:

::

    apt -y install casacore-dev libcfitsio-dev

In order to be able to build the documentation with ``make doc``, ``sphinx`` and some other documentation tools need to be installed:

::

    apt -y install doxygen
    pip3 install sphinx sphinx_rtd_theme breathe myst-parser




Quick installation guide
~~~~~~~~~~~~~~~~~~~~~~~~

::

    git clone --recursive https://git.astron.nl/RD/Radler.git
    cd Radler
    mkdir build && cd build
    cmake -DBUILD_PYTHON_BINDINGS=On ..
    make
    make install


Installation options
~~~~~~~~~~~~~~~~~~~~

(Use :code:`ccmake` or :code:`cmake -i` to configure all options.)

* :code:`BUILD_PYTHON_BINDINGS`: build Python module 'radler' to use Radler from Python
* :code:`BUILD_TESTING`: compile tests

All other build options serve development purposes only, and can/should be left at the default values by a regular user.

All libraries are installed in :code:`<installpath>/lib`. The header files in
:code:`<installpath>/include`. The Python module in
:code:`<installpath>/lib/python{VERSION_MAJOR}.{VERSION_MINOR}/site-packages`. Make sure that your
:code:`LD_LIBRARY_PATH` and :code:`PYTHONPATH` are set as appropiate.
