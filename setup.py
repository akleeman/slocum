"""Slocum: Better forecasts for sailors."""

DOCLINES = __doc__.split("\n")

import sys
import itertools
# multiprocessing isn't directly used, but is require for tests
# https://groups.google.com/forum/#!msg/nose-users/fnJ-kAUbYHQ/_UsLN786ygcJ
import multiprocessing

from setuptools import setup, find_packages

if sys.version_info[:2] < (3, 9):
    raise RuntimeError("Python version 3.9 or higher required.")

MAJOR = 0
MINOR = 1
MICRO = 3
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

if not ISRELEASED:
    VERSION = '%sa0' % VERSION

# https://software.ecmwf.int/wiki/display/GRIB/Python+package+gribapi#_details
requires = ['xarray >= 2022.3',
            'pyproj >= 3.0',
            'pandas >= 1.3',
            'matplotlib >= 3.5',
            'cftime >= 1.5',
            'joblib >= 1.0',
            'retrying >= 1.3',
            'basemap',
            'requests >= 2.25']

setup(name='slocum',
      version=VERSION,
      description="Slocum -- A tool for getting smaller better forecasts to sailors",
      url='http://github.com/akleeman/slocum',
      author='Alex Kleeman and Markus Schweitzer',
      author_email='akleeman@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=requires,
      tests_require=['pytest', 'pytest-coverage'],
      zip_safe=False,
      entry_points={'console_scripts': ['slocum=slocum.run:main']},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        ])
