"""
Poseidon - The god of the sea.

This contains tools for accessing data about the future/past state of the seas.
"""
from __future__ import with_statement

import os
import copy
import numpy as np
import logging
import urllib
import urllib2
import urlparse

from BeautifulSoup import BeautifulSoup

from sl.lib import conventions
from sl.objects import units
from polyglot import Dataset

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

_sources = {'gefs': 'http://motherlode.ucar.edu/thredds/catalog/grib/NCEP/GEFS/Global_1p0deg_Ensemble/members/files/latest.html',
            'nww3': 'http://motherlode.ucar.edu/thredds/catalog/grib/NCEP/WW3/Global/files/latest.html',
            'gfs': 'http://motherlode.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p5deg/files/latest.html',
            }

def latest(latest_html_url):
    f = urllib2.urlopen(latest_html_url)
    soup = BeautifulSoup(f.read())
    def is_latest(x):
        # checks if the beautiful soup a tag holds the latest href
        text = x.fetchText()
        return len(text) == 1 and 'Latest' in str(text[0])
    atag = [x for x in soup.findAll("a") if is_latest(x)]

    if len(atag) != 1:
        raise ValueError("Expected at least one tag with Latest in the name:" +
                         "instead got %s" % str(atag))
    atag = atag[0]
    (_, query) = urllib.splitquery(atag.get(u'href'))
    query = dict(urlparse.parse_qsl(urllib.splitquery(atag.get('href'))[1]))
    dataset = query['dataset']
    return os.path.join('http://motherlode.ucar.edu/thredds/dodsC', dataset)

def subset(nc, ll_crnr, ur_crnr, slicers=None):
    """
    Finds the latest forecast on a netcdf subset server
    """
    ur, ll = ensure_corners(ur_crnr, ll_crnr)

    lats = nc.variables['lat'][:]
    inds = np.nonzero(np.logical_and(lats >= ll.lat, lats <= ur.lat))[0]
    lat_slice = slice(np.min(inds), np.max(inds) + 1)

    lons = nc.variables['lon'][:]
    inds = np.nonzero(np.logical_and(lons >= ll.lon, lons <= ur.lon))[0]
    lon_slice = slice(np.min(inds), np.max(inds) + 1)

    slicers = {} if slicers is None else slicers
    slicers.update({'lat':lat_slice, 'lon':lon_slice})
    out = nc.views(slicers)
    return out

def forecast(source):
    latest_opendap = latest(_sources[source])
    logger.debug(latest_opendap)
    return Dataset(latest_opendap)

def gefs(ll, ur):
    """
    Global Ensemble Forecast System forecast object
    """
    vars = {'u-component_of_wind_height_above_ground':conventions.UWND,
            'v-component_of_wind_height_above_ground':conventions.VWND,}
    fcst = forecast('gefs')
    fcst = fcst.select(vars, view=True)
    fcst = subset(fcst, ll, ur)
    renames = vars
    renames.update(dict((d, conventions.ENSEMBLE) for d in fcst.dimensions if d.startswith('ens')))
    renames.update(dict((d, conventions.TIME) for d in fcst.dimensions if d.startswith('time')))
    renames.update({'lat': conventions.LAT,
                    'lon': conventions.LON})
    fcst = fcst.renamed(renames)
    new_units = fcst['time'].attributes['units'].replace('Hour', 'hours')
    new_units = new_units.replace('T', ' ')
    new_units = new_units.replace('Z', '')
    fcst['time'].attributes['units'] = new_units
    units.normalize_units(fcst[conventions.UWND])
    units.normalize_units(fcst[conventions.VWND])
    return fcst

def gfs(ll, ur):
    """
    Global Ensemble Forecast System forecast object
    """
    vars = {'u-component_of_wind_height_above_ground':conventions.UWND,
            'v-component_of_wind_height_above_ground':conventions.VWND,}
    fcst = forecast('gfs')
    fcst = fcst.select(vars, view=True)
    # subset out the 10m wind height
    ind = np.nonzero(fcst['height_above_ground4'].data[:] == 10.)[0][0]
    fcst = subset(fcst, ll, ur, slicers={'height_above_ground4':slice(ind, ind + 1)})
    # Remove the height above ground dimension
    fcst = copy.deepcopy(fcst)
    fcst = fcst.squeeze(dimension='height_above_ground4')
    renames = vars
    renames.update(dict((d, conventions.TIME) for d in fcst.dimensions if d.startswith('time')))
    renames.update({'lat': conventions.LAT,
                    'lon': conventions.LON})
    fcst = fcst.renamed(renames)
    new_units = fcst['time'].attributes['units'].replace('Hour', 'hours')
    new_units = new_units.replace('T', ' ')
    new_units = new_units.replace('Z', '')
    fcst['time'].attributes['units'] = new_units
    units.normalize_units(fcst[conventions.UWND])
    units.normalize_units(fcst[conventions.VWND])
    return fcst

def ensure_corners(ur, ll, expand=True):
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
    if expand:
        ur.lat += 1
        ur.lon += 1
        ll.lat -= 1
        ll.lon -= 1
    return ur, ll
