"""Sphinx configuration."""
project = "Napari Travali2"
author = "Yohsuke T. Fukai"
copyright = "2025, Yohsuke T. Fukai"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
