"""Slocum: Tools for getting better forecasts to sailors.

A set of tools for serving ultra-compressed weather forecasts.  Includes
 an email-based forecats request service, compression utilities, and
 visualization/decompression tools.
"""

DOCLINES = __doc__.split("\n")

import os
import sys
import itertools
# multiprocessing isn't directly used, but is require for tests
# https://groups.google.com/forum/#!msg/nose-users/fnJ-kAUbYHQ/_UsLN786ygcJ
import multiprocessing

try:
    from setuptools import setup
except ImportError:
    try:
        from setuptools.core import setup
    except ImportError:
        from distutils.core import setup

if sys.version_info[:2] < (2, 6):
    raise RuntimeError("Python version 2.6, 2.7 required.")

MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# https://software.ecmwf.int/wiki/display/GRIB/Python+package+gribapi#_details
requires = {'grib': ['gribapi'],
            'gridded': ['xray == 0.3.1',
                        'pyproj >= 1.9.3',
                        'pandas >= 0.13.1',
                        'matplotlib >= 1.2.0']}
requires['full'] = list(set(itertools.chain(*requires.values())))

setup(name='slocum',
      version='0.1',
      description="Slocum -- A tool for getting smaller better forecasts to sailors",
      url='http://github.com/akleeman/slocum',
      author='Alex Kleeman and Markus Schweitzer',
      author_email='akleeman@gmail.com',
      license='MIT',
      packages=['sl'],
      install_requires=requires['gridded'],
      tests_require=['nose >= 1.0'],
      test_suite='nose.collector',
      zip_safe=False)