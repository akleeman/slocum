from __future__ import with_statement

import os
import re
import uuid
import numpy as np
import coards
import hashlib
import logging
import urllib2
import operator
import datetime
import itertools

from lib import objects, pupynere, iterlib

_data_dir = os.path.join(os.path.dirname(__file__), '../../data/')
_sources = {'ccmp_daily':'ccmp/mean_wind_%Y%m%d_v11l30flk.nc',
            'gefs': 'http://motherlode.ucar.edu:9080/thredds/ncss/grid/NCEP/GEFS/Global_1p0deg_Ensemble/member/GEFS_Global_1p0deg_Ensemble_%Y%m%d_0600.grib2',
            'nww3': 'http://motherlode.ucar.edu:9080/thredds/ncss/grid/fmrc/NCEP/WW3/Global/files/WW3_Global_20110103_1800.grib2/dataset.html',
}
_ibtracs = 'ibtracs/Allstorms.ibtracs_wmo.v03r02.nc'
_storms = 'historical_storms.nc'

def forecast_weather(start_date, ur, ll):
    """
    Yields an iterator of forecast weather as a list each element of which is
    a dictionary mapping from {time:{variable:data_field}} to be used
    during passage simulation.
    """
    ur = ur.copy()
    ll = ll.copy()
    if ur.lon < ll.lon:
        tmp = ur.lon
        ur.lon = ll.lon
        ll.lon = tmp
    if ur.lat < ll.lat:
        tmp = ur.lat
        ur.lat = ll.lat
        ll.lat = tmp
    ur.lat += 1
    ur.lon += 1
    ll.lat -= 1
    ll.lon -= 1

    nww3 = NWW3(ur=ur, ll=ll)
    nww3_fcst = dict(list(nww3.iterator(start_date))[0])

    gefs = GEFS(ur=ur, ll=ll)
    gefs_fcsts = gefs.iterator(start_date)

    # combine the two forecasts
    for fcst in gefs_fcsts:
        fcst = dict(list(fcst))
        for time, variables in fcst.items():
            if time in nww3_fcst:
                variables.update(nww3_fcst[time])
        yield fcst

def historical_weather(start_date, data_dir=None, source='ccmp_daily'):
    if not data_dir:
        data_dir = _data_dir
    fn_format = os.path.join(data_dir, _sources[source])
    storms_path = os.path.join(data_dir, _storms)
    if not os.path.exists(storms_path):
        msg = "Missing data file: %s, Have you created the storms file yet?"
        raise IOError(msg % storms_path)
    storms_nc = objects.DataObject(filename=storms_path, mode='r')

    def historical_year(year):
        now = start_date.replace(year=year)
        for day in range(365):
            def get_data():
                filename = now.strftime(fn_format)
                if not os.path.exists(filename):
                    msg = "Missing data file: %s, Have you downloaded the data for %s yet?"
                    raise IOError(msg % (filename, source))
                nc = objects.DataObject(filename=filename, mode='r')
                nc = nc.slice('time', 0)

                timevar = nc.variables['time']
                # ccmp_daily is stored in files with one day per file
                assert timevar.shape[0] == 1
                # unfortunately the cdo operation invalidates the coard stored times
                timevar = datetime.datetime.strptime(str(timevar[0]), '%Y%m%d.%f')
                # make sure it the requested day
                assert timevar.strftime('%Y%m%d') == now.strftime('%Y%m%d')

                variables = ['uwnd', 'vwnd']
                yield dict((v, objects.DataField(nc, v)) for v in ['uwnd', 'vwnd'])
            yield now, get_data()
            now = now + datetime.timedelta(days=1)

    return [list(historical_year(year)) for year in range(2000, 2009)]

class NCDFSubsetFetcher():
    """
    An abstract class that allows

    requires attributes:
    vars -- {ncdf_name:slocum_name}
    url -- a format string which gives a url to the data file
    source -- a shortname describing the source
    timevar -- the variable which represents the time dimension
    """
    def __init__(self, ur, ll):
        self.ur = ur
        self.ll = ll

    def fetch(self, ur, ll):
        """
        Finds the latest forecast on a netcdf subset server
        """
        args = {'var':','.join(self.vars.keys()),
                'north':'%.2f' % np.ceil(ur.lat),
                'west':'%.2f' % np.floor(ll.lon),
                'south':'%.2f' % np.floor(ll.lat),
                'east':'%.2f' % np.ceil(ur.lon),
                }
        subs = {
            'source':datetime.datetime.now().strftime(self.url),
            'args':'&'.join(['%s=%s' % (k, v) for k, v in sorted(args.items())]),
            }

        url = "%(source)s?%(args)s" % subs

        logging.info(url)
        logging.error("FORECASTS ARE PEGGED TO THE 0600 FORECAST")

        encoded_name = "%s.nc" % str(uuid.UUID(bytes=hashlib.md5(url).digest()))
        logging.info("dumping forecast to: %s" % encoded_name)

        encoded_path = os.path.join(_data_dir, self.source, encoded_name)
        if not os.path.exists(encoded_path):
            urlf = urllib2.urlopen(url)
            with open(encoded_path, 'wb') as f:
                f.write(urlf.read())

        obj = objects.DataObject(encoded_path, 'r')
        units = obj.variables[self.timevar].units.replace('hour ', 'hours ')
        units = re.sub('([0-9])T([0-9])','\\1 \\2', units)
        units = re.sub('Z$','', units)
        obj.variables[self.timevar].units = units

        # rename some ofthe dims to match our conventions
        variable_map = {self.timevar:'time'}
        variable_map.update(self.vars)
        for var, name in variable_map.items():
            obj.rename(var, name)

        return obj

class GEFS(NCDFSubsetFetcher):
    """
    Global Ensemble Forecast System forecast object
    """
    vars = {'U-component_of_wind_height_above_ground':'uwnd',
            'V-component_of_wind_height_above_ground':'vwnd'}
    url = 'http://motherlode.ucar.edu:9080/thredds/ncss/grid/NCEP/GEFS/Global_1p0deg_Ensemble/member/GEFS_Global_1p0deg_Ensemble_%Y%m%d_0600.grib2'
    source = 'gefs'
    timevar = 'time1'

    def iterator(self, start_date):
        """
        returns a list of generators which generate tuples (time, {var:data_field})
        """
        obj = self.fetch(self.ur, self.ll)
        vars = self.vars.values()
        fcsts = list(obj.iterator(dim='height_above_ground1'))
        assert len(fcsts) == 1
        def nciter(obj):
            for t, obj in obj.iterator('time'):
                t = coards.from_udunits(np.asscalar(t.data), t.units)
                if t >= start_date:
                    yield t, dict((k, objects.DataField(obj, k)) for k in vars)
        return [nciter(x[1]) for x in fcsts[0][1].iterator('ens')]

class NWW3(NCDFSubsetFetcher):
    """
    NOAA's WaveWatch forecast model

    http://polar.ncep.noaa.gov/waves/index2.shtml
    """
    vars = {'Significant_height_of_combined_wind_waves_and_swell':'combined_swell_height',
            'Primary_wave_direction':'primary_wave_direction',
            'Direction_of_wind_waves':'direction_of_wind_waves'}
    url = 'http://motherlode.ucar.edu:8080/thredds/ncss/grid/fmrc/NCEP/WW3/Global/runs/NCEP-WW3-Global_RUN_%Y-%m-%dT06:00:00Z'
    source = 'nww3'
    timevar = 'time'

    def iterator(self, start_date):
        """
        returns a list of generators which generate tuples (time, {var:data_field})
        """
        obj = self.fetch(self.ur, self.ll)
        vars = self.vars.values()
        def nciter(obj):
            for t, obj in obj.iterator('time'):
                t = coards.from_udunits(np.asscalar(t.data), t.units)
                if t >= start_date:
                    yield t, dict((k, objects.DataField(obj, k)) for k in vars)
        return [nciter(obj)]