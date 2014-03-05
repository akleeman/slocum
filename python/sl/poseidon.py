"""
Poseidon - The god of the sea.

This contains tools for accessing data about the future/past state of the seas.
"""
from __future__ import with_statement

import os
import copy
import numpy as np
import urllib
import urllib2
import logging
import urlparse

from BeautifulSoup import BeautifulSoup

from xray import Dataset, open_dataset

from sl.lib.objects import NautAngle

import sl.lib.conventions as conv

from sl.lib import units

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

_sources = {'gefs': 'http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GEFS/Global_1p0deg_Ensemble/members/files/latest.html',
            'nww3': 'http://thredds.ucar.edu/thredds/catalog/grib/NCEP/WW3/Global/files/latest.html',
            'gfs': 'http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p5deg/files/latest.html',
            }


def latest(latest_html_url):
    """
    UCAR's thredds server provides access to the latest forecasts
    using a latest.html web page.  Here we parse that page to
    determine the uri to the openDAP data set containing the most
    recent forecast.
    """
    # create a beatiful soup
    f = urllib2.urlopen(latest_html_url)
    soup = BeautifulSoup(f.read())

    def is_latest(x):
        # checks if the beautiful soup 'a' tag holds the latest href
        text = x.fetchText()
        return len(text) == 1 and 'Latest' in str(text[0])
    # get all the possible href links to the latest forecast
    atag = [x for x in soup.findAll("a") if is_latest(x)]
    # we expect there to be only one match
    if len(atag) != 1:
        raise ValueError("Expected at least one tag with Latest in the name:" +
                         "instead got %s" % str(atag))
    atag = atag[0]
    # pull out the query from the atag
    query = dict(urlparse.parse_qsl(urllib.splitquery(atag.get('href'))[1]))
    dataset = query['dataset']
    # the base directory for openDAP data changes to included suffix dodsC
    return os.path.join('http://thredds.ucar.edu/thredds/dodsC', dataset)


def latitude_slicer(fcst, query):
    lat_delta, _ = query['grid_delta']
    domain = query['domain']
    lats = np.asarray(fcst['latitude'].data, dtype=np.float32)
    # assume latitudes are equally spaced for now
    assert np.unique(np.diff(lats)).size == 1
    native_lat_delta = np.abs(np.unique(np.diff(lats))[0])
    # round to the nearest native stride but make sure we dont hit zero
    lat_stride = max(1, int(np.round(lat_delta / native_lat_delta)))

    dist_north_of_domain = lats - domain['N']
    dist_north_of_domain[lats < domain['N']] = np.nan
    northern_most = np.nanargmin(dist_north_of_domain)

    dist_south_of_domain = lats - domain['S']
    dist_south_of_domain[lats > domain['S']] = np.nan
    southern_most = np.nanargmin(dist_south_of_domain)

    sign = np.sign(southern_most - northern_most)
    # if the difference is not a multiple of the stride
    # we could end up chopping the last grid.  By adding
    # stride - 1 inds to the end we avoid that
    southern_most = southern_most + sign * (lat_stride - 1)
    slicer = slice(northern_most, southern_most, sign * lat_stride)

    assert np.all(lats[slicer] <= domain['N'])
    assert np.all(lats[slicer] >= domain['S'])
    assert np.any(lats[slicer] >= domain['N'])
    assert np.any(lats[slicer] <= domain['S'])
    return slicer


def longitude_slicer(fcst, query):
    _, lon_delta = query['grid_delta']
    domain = query['domain']
    lons = np.asarray(fcst['longitude'].data, dtype=np.float32)

    lons = [NautAngle(l) for l in lons]
    diffs = [x.distance_to(y) for x, y in zip(lons[:-1], lons[1:])]
    # assume longitudes are equally spaced for now
    assert np.unique(diffs).size == 1
    native_lon_delta = np.abs(np.unique(diffs)[0])
    # round to the nearest native stride but make sure we dont hit zero
    lon_stride = max(1, int(np.round(lon_delta / native_lon_delta)))

    west = NautAngle(domain['W'])
    east = NautAngle(domain['E'])

    dist_east_of_domain = np.array([x.distance_to(east) for x in lons])
    dist_east_of_domain[dist_east_of_domain > 0] = np.nan
    eastern_most = np.nanargmin(dist_east_of_domain)

    dist_west_of_domain = np.array([x.distance_to(west) for x in lons])
    dist_west_of_domain[dist_west_of_domain < 0] = np.nan
    western_most = np.nanargmin(dist_west_of_domain)

    sign = np.sign(eastern_most - western_most)
    # if the difference is not a multiple of the stride
    # we could end up chopping the last grid.  By adding
    # stride - 1 inds to the end we avoid that
    eastern_most = eastern_most + sign * (lon_stride - 1)
    slicer = slice(western_most, eastern_most, sign * lon_stride)
    assert np.all(lons[slicer] <= domain['E'])
    assert np.all(lons[slicer] >= domain['W'])
    assert np.any(lons[slicer] >= domain['E'])
    assert np.any(lons[slicer] <= domain['W'])
    return slicer


def subset(nc, query):
    """
    Given a forecast (nc) and corners of a spatial subset
    this function returns the smallest subset of the data
    which fully contains the region
    """

    slicers = {'latitude': latitude_slicer(nc, query),
              'longitude': longitude_slicer(nc, query)}
    # and pull out the dataset.  This is delayed till the
    # end because until this point all the data probably
    # lives on a remote server, so we'd like to download
    # as little as possible.
    return nc.indexed_by(**slicers)


def forecast(source):
    """
    A convenience wrapper which will looked up the uri
    to the latest openDAP dataset for source.
    """
    latest_opendap = latest(_sources[source])
    logger.debug(latest_opendap)
    return open_dataset(latest_opendap)


def gfs(query):
    """
    Global Forecast System forecast object
    """
    variables = {}
    if len(set(['wind']).intersection(query['vars'])):
        variables['u-component_of_wind_height_above_ground'] = conv.UWND
        variables['v-component_of_wind_height_above_ground'] = conv.VWND
    if len(set(['rain', 'precip']).intersection(query['vars'])):
        variables['Precipitation_rate_surface_Mixed_intervals_Average'] = conv.PRECIP
    if len(set(['press', 'pressure', 'mslp']).intersection(query['vars'])):
        variables['Pressure_reduced_to_MSL_msl'] = conv.PRESSURE
    if len(variables) == 0:
        raise ValueError("No valid GFS variables in query")

    north, south, east, west = [NautAngle(query['domain'][d])
                                for d in "NSEW"]
    fcst = forecast('gfs')
    fcst = fcst.select(*variables.keys())
    # subset out the 10m wind height
    logger.debug("Selected out variables")
    ind = np.nonzero(fcst['height_above_ground4'].data[:] == 10.)[0][0]
    logger.debug("found 10m height")
    fcst = subset(fcst, north, south, east, west,
                  slicers={'height_above_ground4': slice(ind, ind + 1)})
    logger.debug("subsetted to the domain")
    # Remove the height above ground dimension
    fcst = fcst.squeeze(dimension='height_above_ground4')

    renames = variables
    renames.update(dict((d, conv.TIME) for d in fcst.dimensions if d.startswith('time')))
    renames.update({'lat': conv.LAT,
                    'lon': conv.LON})
    fcst = fcst.renamed(renames)
    fcst = units.normalize_variables(fcst)
    return fcst
