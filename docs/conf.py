# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  
sys.path.insert(0, str("./_ext"))
# print(os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MESA'
version = '0.1.0'
release = '0.1.0'
copyright = '2024, Daisy Yi Ding, Zeyu Tang, Bokai Zhu'
author = 'Daisy Yi Ding, Zeyu Tang, Bokai Zhu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    "sphinx.ext.intersphinx",
    'nbsphinx',
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_copybutton",
]

intersphinx_mapping = dict(  
    python=("https://docs.python.org/3", None),
    numpy=("https://numpy.org/doc/stable/", None),
    statsmodels=("https://www.statsmodels.org/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    networkx=("https://networkx.org/documentation/stable/", None),
    skimage=("https://scikit-image.org/docs/stable/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
)

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints']
autosummary_generate = True
nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_sphinx = False