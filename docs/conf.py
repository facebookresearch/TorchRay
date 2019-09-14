# conf.py - Sphinx configuration
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import sys


def setup(app):
    app.add_stylesheet('css/equations.css')


author = 'TorchRay Contributors'
copyright = 'TorchRay Contributors'
project = 'TorchRay'
release = 'beta'
version = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

exclude_patterns = ['html']
master_doc = 'index'
pygments_style = None
source_suffix = ['.rst', '.md']

# HTML documentation.
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'analytics_id': '',
    'canonical_url': '',
    'display_version': True,
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,

    # Toc options
    'collapse_navigation': True,
    'includehidden': True,
    'navigation_depth': 4,
    'sticky_navigation': True,
    'titles_only': False
}
html_static_path = ['static']
