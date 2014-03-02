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

_sources = {'gefs': 'http://motherlode.ucar.edu/thredds/catalog/grib/NCEP/GEFS/Global_1p0deg_Ensemble/members/files/latest.html',
            'nww3': 'http://motherlode.ucar.edu/thredds/catalog/grib/NCEP/WW3/Global/files/latest.html',
            'gfs': 'http://motherlode.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p5deg/files/latest.html',
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
        # checks if the beautiful soup a tag holds the latest href
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
    return os.path.join('http://motherlode.ucar.edu/thredds/dodsC', dataset)


def subset(nc, north, south, east, west, slicers=None):
    """
    Given a forecast (nc) and corners of a spatial subset
    this function returns the corresponding subset of data
    that contains the spatial region.
    """
    north, south, east, west = map(
            NautAngle, np.radians([north, south, east, west]))
    assert north.is_north_of(south)
    assert east.is_east_of(west)
    # determine which slice we need for latitude
    lats = nc.variables['lat'][:]
    # Note to Alex: Changed the following such that all lat/lon comparisons
    # will use logic defined in NautAngle in order to keep that logic in one
    # central spot. Note that the comparison operators below will call the
    # corresponding NautAngle comparisons.
    n_inds = [south <= NautAngle(np.radians(lat)) for lat in lats]
    s_inds = [north >= NautAngle(np.radians(lat)) for lat in lats]
    inds = np.nonzero(np.logical_and(n_inds, s_inds))[0]
    lat_slice = slice(np.min(inds), np.max(inds) + 1)
    # determine which slice we need for longitude.  GFS uses longitudes
    # between 0 and 360, but slocum uses -180 to 180.  Depending on if
    # the bounding box stradles greenwich or the dateline we want to
    # prefer one or the other interpretations.
    # Note to Alex: The original logic failed for bounding boxes fully east of
    # Greenwich. Same logic as for lats now applied here. Since were just
    # pulling indices it doesn't matter whether lons are [0,360[ or [-180,180[.
    lons = nc.variables['lon'][:]
    # TODO: check: east == -10 and west == 10 would result in taking the lon
    # slice the 'long way around'
    # if east < 0 and west > 0.:
    #     east = east % 360
    #     west = west % 360
    # else:
    #     lons = np.mod(lons + 180, 360) - 180
    # TODO: need to fix, fails for eastern longitudes (e.g. inds will be an
    # empty array for east = 155 and west = 151
    e_inds = [west <= NautAngle(np.radians(lon)) for lon in lons]
    w_inds = [east >= NautAngle(np.radians(lon)) for lon in lons]
    inds = np.nonzero(np.logical_and(e_inds, w_inds))[0]
    lon_slice = slice(np.min(inds), np.max(inds) + 1)

    # assert np.all(lons[lon_slice] >= west)
    # assert np.all(lons[lon_slice] <= east)
    assert np.all([west <= NautAngle(np.radians(lon))
                   for lon in lons[lon_slice]])
    assert np.all([east >= NautAngle(np.radians(lon))
                   for lon in lons[lon_slice]])
    
    if east >= 0 and west < 0:
        # sorry brits, this is going to take a while
        rhs = subset(nc, north, south, 0., east, slicers=slicers)
        lhs = subset(nc, north, south, west, 0., slicers=slicers)
        # TODO, finish splicing objects.
        raise ValueError("sorry brits.")
    # add lat/lon slicers to any additional slicers
    slicers = {} if slicers is None else slicers
    slicers.update({'lat': lat_slice, 'lon': lon_slice})
    # and pull out the dataset.  This is delayed till the
    # end because until this point all the data probably
    # lives on a remote server, so we'd like to download
    # as little as possible.
    # Note to Alex: replaced views (no longer in Dataset) with indexed_by;
    # eventually, the whole subsetting would be leaner if we used
    # labeled_by...someday...
    out = nc.indexed_by(slicers)
    return out


def forecast(source):
    """
    A convenience wrapper which will looked up the uri
    to the latest openDAP dataset for source.
    """
    latest_opendap = latest(_sources[source])
    logger.debug(latest_opendap)
    return open_dataset(latest_opendap)


def gefs(ll, ur):
    """
    Global Ensemble Forecast System forecast object
    """
    vars = {'u-component_of_wind_height_above_ground': conv.UWND,
            'v-component_of_wind_height_above_ground': conv.VWND,}
    fcst = forecast('gefs')
    fcst = fcst.select(*vars.keys())
    # TODO: fix signature and call to subset
    fcst = subset(fcst, ll, ur)
    renames = vars
    renames.update(dict((d, conv.ENSEMBLE) for d in fcst.dimensions if d.startswith('ens')))
    renames.update(dict((d, conv.TIME) for d in fcst.dimensions if d.startswith('time')))
    renames.update({'lat': conv.LAT,
                    'lon': conv.LON})
    fcst = fcst.renamed(renames)
    new_units = fcst['time'].attributes['units'].replace('Hour', 'hours')
    new_units = new_units.replace('T', ' ')
    new_units = new_units.replace('Z', '')
    fcst['time'].attributes['units'] = new_units
    units.normalize_units(fcst[conv.UWND])
    units.normalize_units(fcst[conv.VWND])
    return fcst


def gfs(north, south, east, west, variables=None):
    """
    Global Forecast System forecast object
    """
    if variables is None:
        variables = {'u-component_of_wind_height_above_ground': conv.UWND,
                     'v-component_of_wind_height_above_ground': conv.VWND,}
    fcst = forecast('gfs')
    fcst = fcst.select(*variables.keys())
    # subset out the 10m wind height
    logger.debug("Selected out variables")
    ind = np.nonzero(fcst['height_above_ground4'].data[:] == 10.)[0][0]
    logger.debug("found 10m height")
    # Note to Alex: Changed call to subset to match new signature with north,
    # south, west, east
    # fcst = subset(fcst, ll, ur, slicers={'height_above_ground4': slice(ind, ind + 1)})
    fcst = subset(
            fcst, north, south, east, west,
            slicers={'height_above_ground4': slice(ind, ind + 1)})
    logger.debug("subsetted to the domain")
    # Remove the height above ground dimension
    fcst = copy.deepcopy(fcst)
    fcst = fcst.squeeze(dimension='height_above_ground4')
    renames = variables
    # was there a specific reason for the 'special treatment' of the tme
    # dimenesion?
    renames.update(dict((d, conv.TIME) for d in fcst.dimensions if d.startswith('time')))
    renames.update({'lat': conv.LAT,
                    'lon': conv.LON})
    fcst = fcst.renamed(renames)
    # xray time indices are datetime subclasses, no mangling required
    # new_units = fcst['time'].attributes['units'].replace('Hour', 'hours')
    # new_units = new_units.replace('T', ' ')
    # new_units = new_units.replace('Z', '')
    # fcst['time'].attributes['units'] = new_units
    fcst = units.normalize_variables(fcst)
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
