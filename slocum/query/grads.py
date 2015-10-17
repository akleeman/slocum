import xray
import joblib
import urllib2
import logging
import datetime
import retrying
import requests

import numpy as np
import pandas as pd

import variables

logging.basicConfig(level=logging.DEBUG)


def recent_datasets(url_format, freq_hours=None, n=None):
    """
    Returns a list of the urls to the 'n' most recent datasets that have
    urls with the provided format.  This is done by iterating backwards
    in time using the frequency provided.
    """
    # default to six hour frequency
    freq_hours = freq_hours or 6
    n = n or 10
    # figure out the current time UTC plus a buffer
    now = datetime.datetime.utcnow()
    now = now + datetime.timedelta(hours=1)
    # round to the nearest freq hour
    start = datetime.datetime(now.year, now.month, now.day,
                              now.hour - np.mod(now.hour, freq_hours))
    # list all the times
    times = pd.date_range(start, periods=n, freq='-%dH' % freq_hours)
    # format all the times
    return [datetime.datetime.strftime(t, url_format) for t in times]


def is_url_error(exception):
    """ Used to catch UrlErrors in retrying logic """
    return isinstance(exception, urllib2.URLError)


@retrying.retry(retry_on_exception=is_url_error,
                wait_random_min=100, wait_random_max=2000,
                stop_max_attempt_number=3)
def opendap_exists(url):
    """
    Attempts to open a GrADS opendap url and inspects the response to
    decide if the url was a valid dataset or not.
    """
    f = urllib2.urlopen(url)
    # unfortunately opendap servers will respond even if the dataset
    # exists, so we need to search the html source to decide if the
    # server failed or not.  When a GrADS server fails it says
    # GrADS Data Server - error in the first few lines
    exists = not 'error' in f.read(200)
    if not exists:
        logging.debug("Dataset not found: %s" % url)
    return exists


def first(lst, key=lambda x:x):
    """
    Returns the first item that is True, or optionally, for which
    key(element) is True.
    """
    for l in lst:
        if key(l):
            return l


def most_recent_dataset(url_format, freq_hours, n=None):
    """
    This attempts to open the 'n' most recent urls for a given dataset
    and returns the most recent one that exists.  If none exist a
    ValueError is raised.
    """
    datasets = recent_datasets(url_format, freq_hours, n)
    exists = joblib.Parallel(4)(joblib.delayed(opendap_exists)(x)
                                for x in datasets)
    if not any(exists):
        raise ValueError("Unable to find any recent datasets. format: %s"
                         % url_format)
    # return the first dataset that exists, since the datasets
    # are in reverse chronological order that should be the most recent
    most_recent = first(zip(exists, datasets), key=lambda x:x[0])[1]
    logging.debug("Most recent dataset: %s" % most_recent)
    return most_recent


class GrADS(object):
    """
    An abstract object which takes care of the standard normalization
    and processing involved in fetching a dataset from a GrADS server.
    """

    # this must be defined by implementing classes.
    url_format = None
    frequency = 6

    grads_names = {'ugrd10m': 'x_wind',
                   'vgrd10m': 'y_wind',
                   'tmp2m': 'air_temperature',
                   'ens': 'realization',
                   'lat': 'latitude',
                   'lon': 'longitude',
                   'prmslmsl': 'air_pressure_at_sea_level',
                   'u_velocity': 'sea_water_x_velocity',
                   'v_velocity': 'sea_water_y_velocity',
                   'sst': 'sea_surface_temperature',
                   'htsgwsfc': 'sea_surface_wave_significant_height',
                   'dirpwsfc': 'sea_surface_wave_to_direction'
                   }

    grads_units = {'x_wind': 'm/s',
                   'y_wind': 'm/s',
                   'sea_surface_temperature': 'c',
                   'latitude': 'degrees_north',
                   'longitude': 'degrees_east',
                   'air_temperature': 'c',
                   'sea_water_x_velocity': 'm/s',
                   'sea_water_y_velocity': 'm/s',
                   'sea_surface_wave_significant_height': 'm',
                   'sea_surface_wave_to_direction': 'degrees'
                   }

    def variables(self):
        """
        Returns a list of variable objects that are contained within a
        particular dataset.  These should be variables similar to those
        found in slocum.query.variables.
        """
        raise NotImplementedError("variables() is not implemented")

    def fetch(self, url=None):
        """
        Fetches either the most recent forecast (if url is None) or
        the forecast specified by the url.
        """
        if url is None:
            url = most_recent_dataset(self.url_format, self.frequency)
        ds = xray.open_dataset(url)
        return self.normalize(ds)

    def normalize(self, ds):
        """
        Nudges the dataset towards CF conventions.
s        """
        # determine which variables need to be renamed
        renames = {k: v for k, v in self.grads_names.iteritems()
                   if k in ds}
        ds = ds.rename(renames)
        # GrADS uses 1 indexing for ensemble realizations, we prefer 0.
        if 'realization' in ds:
            ds['realization'] = ('realization',
                                 np.arange(ds['realization'].size),
                                 )
        # grads doesn't follow cf conventions for units, so we
        # need to have them hard coded.
        for k, v in ds.variables.iteritems():
            if k in self.grads_units:
                v.attrs['units'] = self.grads_units[k]
        # grads uses time units of days, with float values, this
        # also violates cf so we switch to hour units
        ref_time = pd.to_datetime(ds['time'].values[0])
        new_units = ref_time.strftime('hours since %Y-%m-%d %H:%M:%SZ')
        ds['time'].encoding['units'] = new_units
        # extract only the variables that we recognize
        return ds[[x for x in self.grads_names.values()
                   if x in ds]]


class GEFS(GrADS):
    """
    The Global Ensemble Forecast System
    """

    url_format = 'http://nomads.ncep.noaa.gov:9090/dods/gens_bc/gens%Y%m%d/gep_all_%Hz'
    freq = 6

    def variables(self):
        return [variables.wind,
                variables.temperature,
                variables.pressure]


class CMCENS(GrADS):
    """
    The Canadian Ensemble Forecasting system
    """
    url_format = 'http://nomads.ncep.noaa.gov:9090/dods/cmcens/cmcens%Y%m%d/cmcens_all_%Hz'
    freq = 12

    def variables(self):
        return [variables.wind,
                variables.temperature,
                variables.pressure]


class FENS(GrADS):
    """
    The FNMOC Ensemble Forecast System
    """
    url_format = 'http://nomads.ncep.noaa.gov:9090/dods/fens/fens%Y%m%d/fens_all_%Hz'
    freq = 6

    def variables(self):
        return [variables.wind,
                variables.temperature,
                variables.pressure]


class GFS(GrADS):
    """
    Fetches from the quarter degree Global Forecast System
    """
    url_format = 'http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs%Y%m%d/gfs_0p25_%Hz'
    freq = 6

    def variables(self):
        return [variables.wind,
                variables.temperature,
                variables.pressure]


class RTOFS(GrADS):
    """
    The Real Time Ocean Forecast System
    """
    url_format = 'http://nomads.ncep.noaa.gov:9090/dods/rtofs/rtofs_global%Y%m%d/rtofs_glo_2ds_forecast_3hrly_prog'
    freq = 24

    def normalize(self, ds):
        ds = ds.squeeze('lev')
        return super(RTOFS, self).normalize(ds)

    def variables(self):
        return [variables.current,
                variables.sea_surface_temperature]


class WW3(GrADS):

    url_format = 'http://nomads.ncep.noaa.gov:9090/dods/wave/nww3/nww3%Y%m%d/nww3%Y%m%d_%Hz'
    freq = 6

    def variables(self):
        return [variables.wave_height,
                variables.wave_direction]

