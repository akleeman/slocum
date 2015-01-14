"""Slocum: Better forecasts for sailors."""

DOCLINES = __doc__.split("\n")

import sys
import itertools
# multiprocessing isn't directly used, but is require for tests
# https://groups.google.com/forum/#!msg/nose-users/fnJ-kAUbYHQ/_UsLN786ygcJ
import multiprocessing

from setuptools import setup, find_packages

if sys.version_info[:2] < (2, 6):
    raise RuntimeError("Python version 2.6, 2.7 required.")

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

if not ISRELEASED:
    VERSION = '%sa0' % VERSION

# https://software.ecmwf.int/wiki/display/GRIB/Python+package+gribapi#_details
requires = {'grib': ['gribapi'],
            'gridded': ['xray == 0.3.1',
                        'pyproj >= 1.9.3',
                        'pandas >= 0.13.1',
                        'matplotlib >= 1.2.0',
                        'netcdf4',
                        'basemap']}
requires['full'] = list(set(itertools.chain(*requires.values())))

setup(name='slocum',
      version=VERSION,
      description="Slocum -- A tool for getting smaller better forecasts to sailors",
      url='http://github.com/akleeman/slocum',
      author='Alex Kleeman and Markus Schweitzer',
      author_email='akleeman@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=requires['gridded'],
      tests_require=['nose >= 1.0'],
      test_suite='nose.collector',
      zip_safe=False,
    entry_points={'console_scripts': ['slocum=slocum.run:main']},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        ])
