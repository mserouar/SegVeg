# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'yellowgreen-multi'
copyright = "2021, Hiphen"
author = 'Hiphen'

import yellowgreenmulti
# The full version, including alpha/beta/rc tags
release = yellowgreenmulti.__version__
# The short X.Y version
version = '.'.join(release.split('.')[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'm2r2',
	'sphinx.ext.autodoc',
	'sphinx.ext.intersphinx',
	'sphinx.ext.todo',
	'sphinxcontrib.autoprogram'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
	'collapse_navigation': False,
	'navigation_depth': 4,
	'prev_next_buttons_location': 'both',
	'titles_only': False
	}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, the reST sources are included in the HTML build as _sources/name.
html_copy_source = False

# If true, generate domain-specific indices in addition to the general index. For e.g. the Python
# domain, this is the global module index.
html_domain_indices = False

# If true, add an index to the HTML documents.
html_use_index = False

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# -- Options for autodoc extension ---------------------------------------
autoclass_content = 'both'
autodoc_member_order = 'groupwise'

# -- Options for todo extension ---------------------------------------
todo_include_todos = True


# -- Setup ---------------------------------------------------------------

def setup(app):
	app.add_css_file('theme_overrides.css')
