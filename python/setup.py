# multiprocessing isn't directly used, but is require for tests
# https://groups.google.com/forum/#!msg/nose-users/fnJ-kAUbYHQ/_UsLN786ygcJ
import multiprocessing
from setuptools import setup

setup(name='slocum',
      version='0.1',
      description="Slocum -- A tool for getting smaller better forecasts to sailors",
      url='http://github.com/akleeman/slocum',
      author='Alex Kleeman and Markus Schweitzer',
      author_email='akleeman@gmail.com',
      license='MIT',
      packages=['sl'],
      install_requires=[
          'BeautifulSoup>=3.2.0',
          'netCDF4 >= 1.0.6',
          'numpy >= 1.8',
          'pandas >= 0.13.1'
          'pyproj>=1.9.3',
          'xray>=0.0.0',
      ],
      dependency_links=['https://github.com/akleeman/xray/zipball/master#egg=xray-0.0.0'],
      extra_requires={'grib': 'gribapi'},
      tests_require=['nose >= 1.0'],
      test_suite='nose.collector',
      zip_safe=False)