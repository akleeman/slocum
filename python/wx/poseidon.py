"""
Poseidon - The god of the sea.

This contains tools for accessing data about the future/past state of the seas.
"""
from __future__ import with_statement

import os
import re
import uuid
import numpy as np
import hashlib
import logging
import urllib2
import datetime

import wx.lib.conventions as conv
from wx.objects import objects, units, core

_data_dir = os.path.join(os.path.dirname(__file__), '../../data/')
_sources = {'ccmp_daily':'ccmp/mean_wind_%Y%m%d_v11l30flk.nc',
            'gefs': 'http://motherlode.ucar.edu:9080/thredds/ncss/grid/NCEP/GEFS/Global_1p0deg_Ensemble/member/GEFS_Global_1p0deg_Ensemble_%Y%m%d_0600.grib2',
            'nww3': 'http://motherlode.ucar.edu:9080/thredds/ncss/grid/fmrc/NCEP/WW3/Global/files/WW3_Global_20110103_1800.grib2/dataset.html',
}
_ibtracs = 'ibtracs/Allstorms.ibtracs_wmo.v03r02.nc'
_storms = 'historical_storms.nc'

def ncdf_subset(url, ll, ur, vars):
    """
    Finds the latest forecast on a netcdf subset server
    """
    ur, ll = ensure_corners(ur, ll)

    args = {'var':','.join(vars.keys()),
            'north':'%.2f' % np.ceil(ur.lat),
            'west':'%.2f' % np.floor(ll.lon),
            'south':'%.2f' % np.floor(ll.lat),
            'east':'%.2f' % np.ceil(ur.lon),
            }

    subs = {'source':datetime.datetime.now().strftime(url),
            'args':'&'.join(['%s=%s' % (k, v) for k, v in sorted(args.items())]),
            }

    url = "%(source)s?%(args)s" % subs

    logging.info(url)
    logging.error("FORECASTS ARE PEGGED TO THE 0600 FORECAST")

    encoded_name = "%s.nc" % str(uuid.UUID(bytes=hashlib.md5(url).digest()))
    logging.info("dumping forecast to: %s" % encoded_name)

    encoded_path = os.path.join(_data_dir, 'ncdf_subset', encoded_name)
    if not os.path.exists(encoded_path):
        urlf = urllib2.urlopen(url)
        with open(encoded_path, 'wb') as f:
            f.write(urlf.read())

    obj = objects.Data(encoded_path)
    return units.normalize_data(obj.renamed(**vars))

def gefs_subset(ll, ur):
    """
    Global Ensemble Forecast System forecast object
    """
    vars = {'U-component_of_wind_height_above_ground':'uwnd',
            'V-component_of_wind_height_above_ground':'vwnd'}
    url = 'http://motherlode.ucar.edu:9080/thredds/ncss/grid/NCEP/GEFS/Global_1p0deg_Ensemble/member/GEFS_Global_1p0deg_Ensemble_%Y%m%d_0600.grib2'

    obj = ncdf_subset(url=url, ll=ll, ur=ur, vars=vars)
    obj = obj.renamed(time1=conv.TIME, ens=conv.ENSEMBLE)
    obj = obj.squeeze(dimension='height_above_ground1')
    return obj

def nww3_subset(ll, ur):
    vars = {'Significant_height_of_combined_wind_waves_and_swell':'combined_swell_height',
            'U-component_of_wind':'uwnd',
            'V-component_of_wind':'vwnd',
            'Primary_wave_direction':'primary_wave_direction',
            'Direction_of_wind_waves':'direction_of_wind_waves'}
    url = 'http://motherlode.ucar.edu:8080/thredds/ncss/grid/fmrc/NCEP/WW3/Global/runs/NCEP-WW3-Global_RUN_%Y-%m-%dT06:00:00Z'

    obj = ncdf_subset(url=url, ll=ll, ur=ur, vars=vars)
    import pdb; pdb.set_trace()
    return obj


def ensure_corners(ur, ll):
    """
    Makes sure the upper right (ur) corner is actually the upper right.  Same
    for the lower left (ll).
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
    return ur, ll

def forecast_weather(start_date, ur, ll):
    """
    Yields an iterator of forecast weather as a list each element of which is
    a dictionary mapping from {time:{variable:data_field}} to be used
    during passage simulation.
    """
    ur, ll = ensure_corners(ur, ll)

    gefs = gefs_subset(ur=ur, ll=ll)
    nww3 = nww3_subset(ur=ur, ll=ll)
    return [gefs, nww3]

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

#
#    def fetch_opendap(self, ur=None, ll=None):
#        """
#        Finds the latest forecast on a netcdf subset server
#        """
#        if ur is None:
#            ur = self.ur
#        if ll is None:
#            ll = self.ll
#
#        args = {'var':','.join(self.vars.keys()),
#                'north':'%.2f' % np.ceil(ur.lat),
#                'west':'%.2f' % np.floor(ll.lon),
#                'south':'%.2f' % np.floor(ll.lat),
#                'east':'%.2f' % np.ceil(ur.lon),
#                }
#
#        from pydap.client import open_url
#        import pydap.lib
#        pydap.lib.CACHE = "/tmp/pydap-cache/"
#
#        dataset = open_url(self.opendap)
#
#        lat = dataset['lat'][:]
#        lon = dataset['lon'][:]
#        uwnd = dataset['ugrd10m']
#        vwnd = dataset['vgrd10m']
#
#        slicer = [slice(None, None, None)] * len(uwnd.dimensions)
#
#        lat_low = np.min(np.where(lat < ll.lat))
#        lat_high = np.max(np.where(lat < ur.lat))
#        slicer[list(uwnd.dimensions).index('lat')] = slice(lat_low, lat_high, 1)
#
#        #blah = open_url('http://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod/wave.20110124/gep04.glo_60m.t00z.grib2')
#        blah = open_url('%s?ugrd10m[0:1:20][0:1:64][80:1:90][200:1:210]' % self.opendap)
#
#        import pdb; pdb.set_trace()
#        tmp = uwnd[slicer]
#
#        obj = objects.DataObject(encoded_path, 'r')
#        units = obj.variables[self.timevar].units.replace('hour ', 'hours ')
#        units = re.sub('([0-9])T([0-9])','\\1 \\2', units)
#        units = re.sub('Z$','', units)
#        obj.variables[self.timevar].units = units
#
#        # rename some ofthe dims to match our conventions
#        variable_map = {self.timevar:'time'}
#        variable_map.update(self.vars)
#        for var, name in variable_map.items():
#            obj.rename(var, name)
#
#        return obj