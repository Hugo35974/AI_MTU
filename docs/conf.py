import os
import sys

sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../contener'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'Electricity Forecasting'
copyright = '2024, Hugo Bénard'
author = 'Hugo Bénard'
release = '0.1.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # For automatic documentation generation
    'sphinx.ext.napoleon', # For Google and NumPy style docstrings
    'sphinx.ext.viewcode', # To include links to the source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
source_suffix = ['.rst']
