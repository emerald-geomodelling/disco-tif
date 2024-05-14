#!/usr/bin/env python

import os, sys
from setuptools import setup, find_packages

setup(name='disco-tif',
      version="0.0.3",
      url='https://github.com/emerald-geomodelling/disco-tif',
      author="Benjamin Bloss",
      author_email='bb@emrld.no',
      description="Geotiff processing utilities",
      install_requires=["seaborn",
                        "rasterio==1.3.10",
                        "numpy",
                        "pandas",
                        "matplotlib",
                        "earthpy",
                        "scikit-learn",
                       ],
      long_description="Geotiff processing utilities",
      include_package_data=True,
      keywords=['raster_processing', 'EMerald_custom_colormap', 'hillshade', 'pca'],
      license='MIT',
      packages=find_packages(),
     )
