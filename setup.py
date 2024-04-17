#!/usr/bin/env python

import os,sys
from setuptools import setup, find_packages

setup(name = 'disco_tif',
      version = "0.0.1",
      url = 'https://github.com/emerald-geomodelling/disco_tif',
      author = "Benjamin Bloss",
      author_email = 'bb@emrld.no',
      description = "Geotiff processing utilities",
      install_requires = ["seaborn",
                          "rasterio",
                          "numpy",
                          "pandas",
                          "matplotlib",
                          "earthpy",
                          "scikit-learn",
                         ],
      long_description = "Geotiff processing utilities",
      include_package_data = True,
      keywords = 'raster_processing',
      packages = find_packages(),
     )
