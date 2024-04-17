#!/usr/bin/env python

import os,sys
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = ["seaborn",
                "rasterio",
                "numpy",
                "pandas",
                "matplotlib",
                "earthpy",
                "scikit-learn",
               ]

setup(
    author="Benjamin Bloss",
    author_email='bb@emrld.no',
    # python_requires='>=3.10',
    description="Geotiff processing utilities",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='raster_processing',
    name='disco_tif',
    packages=find_packages(),
    url='https://github.com/emerald-geomodelling/disco_tif',
    version='0.0.1',
    # zip_safe=False,
)
