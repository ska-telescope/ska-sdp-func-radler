# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

# Make sure that it refers to the shared object file from current build
if "READTHEDOCS" in os.environ:
    sys.path.insert(0, os.path.abspath("../build/python"))
else:
    sys.path.insert(0, os.environ["RADLER_SO_PATH"])

# Give informative error on not finding radler, rather than giving this
# information only in the generated HTML documentation
try:
    import radler
except ModuleNotFoundError:
    raise RuntimeError(f"Radler not found at {sys.path[0]}")

# -- Project information -----------------------------------------------------

project = "Radler"
copyright = "%d, André Offringa" % date.today().year
author = "André Offringa"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "breathe",
]

# Disable typehints in signatures - doens't seem to take any effect
autodoc_typehints = "none"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

# Breathe Configuration
# When using CMake, the 'doc' target already sets breathe_projects.
if "READTHEDOCS" in os.environ:
    breathe_projects = {"Radler": "../build/doc/doxygen/xml"}

# Breathe Configuration
breathe_default_project = "Radler"
