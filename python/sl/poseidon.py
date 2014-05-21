"""
Poseidon - The god of the sea.

This contains tools for accessing data about the future/past state of the seas.
"""
from __future__ import with_statement

import os
import numpy as np
import pandas as pd
import urllib
import urllib2
import logging
import urlparse
import datetime

from BeautifulSoup import BeautifulSoup

import xray

import sl.lib.conventions as conv

from sl.lib import units
from sl.lib.objects import NautAngle

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

_servers = ['http://unidata2-new.ssec.wisc.edu',
            'http://thredds.ucar.edu']

_models = {'gefs': 'NCEP/GEFS/Global_1p0deg_Ensemble/member',
            'nww3': '/NCEP/WW3/Global/files/',
            'gfs': '/NCEP/GFS/Global_0p5deg/member',
            }

_files = {'gfs': 'NCEP/GFS/Global_0p5deg/files/GFS_Global_0p5deg_%Y%m%d_%H00.grib2',
          'gefs': 'NCEP/GEFS/Global_1p0deg_Ensemble/files/Global_1p0deg_Ensemble_%Y%m%d_%H00.grib2'}


def latest_url(model, server):
    return os.path.join(server, 'thredds/catalog/',
                        _models[model], 'latest.html')


def file_format(model, server):
    return os.path.join(server, 'thredds/dodsC/grib/', _files[model])


def best_url(model, server):
    """
    Despite the name, this is often the oldest forecast.
    """
    return os.path.join(server, 'thredds/dodsC/grib', model, 'best.html')


def latest(model, server):
    """
    UCAR's thredds server provides access to the latest forecasts
    using a latest.html web page.  Here we parse that page to
    determine the uri to the openDAP data set containing the most
    recent forecast.
    """
    # create a beatiful soup
    f = urllib2.urlopen(latest_url(model, server))
    soup = BeautifulSoup(f.read())

    def is_grib(x):
        # checks if the beautiful soup 'a' tag holds the latest href
        text = x.fetchText()
        return len(text) == 1 and ('grib2' in str(text[0]) or
                                   'latest' in str(text[0]).lower())
    # get all the possible href links to the latest forecast
    atag = [x for x in soup.findAll("a") if is_grib(x)]
    # we expect there to be only one match
    if len(atag) != 1:
        raise ValueError("Expected at only one link to the latest forecast,"
                         "instead got %s" % str(atag))
    atag = atag[0]
    # pull out the query from the atag
    query = dict(urlparse.parse_qsl(urllib.splitquery(atag.get('href'))[1]))
    dataset = query['dataset']
    # the base directory for openDAP data changes to included suffix dodsC
    return os.path.join(server, 'thredds/dodsC', dataset)


def fallback(model, server):
    # start at an overestimate of the most recent forecast and backtrack
    # until one is found.
    start = datetime.datetime.utcnow().strftime('%Y-%m-%d 18:00')
    for d in pd.date_range(start, periods=12, freq='-6H'):
        try:
            url = d.strftime(file_format(model, server))
            logger.info("Trying to load %s" % url)
            ds = xray.open_dataset(url)
            return ds
        except:
            pass
    try:
        return xray.open_dataset(best_url(model, server))
    except:
        raise ValueError("Could not find a valid forecast. "
                         "Perhaps the server is down?")


def latitude_slicer(lats, query):
    """
    Returns a slice object that will slice out the smallest chunk of lats
    that covers the query domain defined by query['domain']['N'] and
    query['domain']['S'].

    The resulting slice will result in latitudes which descend from north
    to south, with a grid delta that is closest to the request grid delta,
    query['grid_delta'][0].
    """
    domain = query['domain']
    # make sure north is actually north of south.
    assert query['domain']['N'] > query['domain']['S']
    lats = np.asarray(lats, dtype=np.float32)
    # assume latitudes are equally spaced for now
    assert np.unique(np.diff(lats)).size == 1
    native_lat_delta = np.abs(np.unique(np.diff(lats))[0])
    lat_delta, _ = query.get('grid_delta', (native_lat_delta, None))
    # round to the nearest native stride but make sure we dont hit zero
    lat_stride = max(1, int(np.round(lat_delta / native_lat_delta)))

    dist_north_of_domain = lats - domain['N']
    dist_north_of_domain[lats < domain['N']] = np.nan
    northern_most = np.nanargmin(dist_north_of_domain)

    dist_south_of_domain = lats - domain['S']
    dist_south_of_domain[lats > domain['S']] = np.nan
    southern_most = np.nanargmax(dist_south_of_domain)

    sign = np.sign(southern_most - northern_most)
    # if the difference is not a multiple of the stride
    # we could end up chopping the last grid.  By adding
    # stride - 1 inds to the end we avoid that.  We then
    # have to add another + 1 to the slicer to make it inclusive.
    southern_most = southern_most + sign * lat_stride
    slicer = slice(northern_most, southern_most, sign * lat_stride)

#     assert np.all(lats[slicer] <= domain['N'])
#     assert np.all(lats[slicer] >= domain['S'])
#     assert np.any(lats[slicer] >= domain['N'])
#     assert np.any(lats[slicer] <= domain['S'])
    return slicer


def longitude_slicer(lons, query):
    """
    Returns a slice object that will slice out the smallest chunk of lons
    that covers the query domain defined by query['domain']['W'] and
    query['domain']['E'].

    The resulting slice will result in longitudes which increase from west
    to east, with a grid delta that is closest to the request grid delta,
    query['grid_delta'][1].
    """
    domain = query['domain']
    lons = np.asarray(lons, dtype=np.float32)

    lons = [NautAngle(l) for l in lons]
    diffs = [x.distance_to(y) for x, y in zip(lons[:-1], lons[1:])]
    # assume longitudes are equally spaced for now
    assert np.unique(diffs).size == 1
    native_lon_delta = np.abs(np.unique(diffs)[0])
    _, lon_delta = query.get('grid_delta', (None, native_lon_delta))
    # round to the nearest native stride but make sure we dont hit zero
    lon_stride = max(1, int(np.round(lon_delta / native_lon_delta)))

    west = NautAngle(domain['W'])
    east = NautAngle(domain['E'])

    assert east.is_east_of(west)

    dist_east_of_domain = np.array([x.distance_to(east) for x in lons])
    dist_east_of_domain[dist_east_of_domain > 0] = np.nan
    eastern_most = np.nanargmax(dist_east_of_domain)

    dist_west_of_domain = np.array([x.distance_to(west) for x in lons])
    dist_west_of_domain[dist_west_of_domain < 0] = np.nan
    western_most = np.nanargmin(dist_west_of_domain)

    sign = np.sign(eastern_most - western_most)
    # if the difference is not a multiple of the stride
    # we could end up chopping the last grid.  By adding
    # stride - 1 inds to the end we avoid that.   We then
    # have to add another + 1 to the slicer to make it inclusive.
    eastern_most = eastern_most + sign * lon_stride
    slicer = slice(western_most, eastern_most, sign * lon_stride)

#     assert np.all([x <= domain['E'] for x in lons[slicer]])
#     assert np.all([x >= domain['W'] for x in lons[slicer]])
#     assert np.any([x >= domain['E'] for x in lons[slicer]])
#     assert np.any([x <= domain['W'] for x in lons[slicer]])

    return slicer


def time_slicer(time_coordinate, query):
    """
    Returns a slice object that will slice out all times
    upto the largest query['hours'].  This allows the first
    slice of the time dimension to be lazy, after which
    subset_time should be used.
    """
    # next step is parsing out the times
    # we assume that the forecast units are in hours
    ref_time = time_coordinate.data[0]
    max_hours = max(query['hours'])
    assert int(max_hours) == max_hours
    max_time = ref_time + np.timedelta64(int(max_hours), 'h')
    max_ind = np.max(np.nonzero(time_coordinate.data <= max_time)[0])
    assert max_ind > 0
    return slice(0, max_ind + 1)


def subset_time(fcst, hours):
    """
    Extracts all the forecast valid times from fcst for lead
    times of 'hours'.  If used with a remote dataset such as
    an openDAP server this will download all the data.  Instead
    consider using time_slicer first.
    """
    # next step is parsing out the times
    ref_time = np.datetime64(fcst[conv.TIME].data[0])
    # we are assuming that the first time is the reference time
    # we can check that by converting back to cf units and making
    # sure that the first cf time is 0.
    ref_time = xray.decode_cf_datetime(0.,
                                fcst['time'].encoding['units'],
                                fcst['time'].encoding.get('calendar', None))
    hours = np.array(hours)
    # make sure hours are all integers
    np.testing.assert_array_almost_equal(hours, hours.astype('int'))
    times = np.array([ref_time + np.timedelta64(int(x), 'h') for x in hours])
    return fcst.labeled_by(time=times)


def subset(remote_dataset, query, additional_slicers=None):
    """
    Given a forecast (nc) and corners of a spatial subset
    this function returns the smallest subset of the data
    which fully contains the region
    """
    slicers = {conv.LAT: latitude_slicer(remote_dataset[conv.LAT], query),
               conv.LON: longitude_slicer(remote_dataset[conv.LON], query),
               conv.TIME: time_slicer(remote_dataset[conv.TIME], query)}
    if not additional_slicers is None:
        slicers.update(additional_slicers)
    # Until this point all the data might live on a remote server,
    # so we'd like to download as little as possible.  As a result
    # we split the subsetting into two steps, the first can be done
    # using slicers which minimizes downloading from openDAP servers,
    # the second pulls out the actual requested domain once the data
    # has been loaded locally.
    local_dataset = remote_dataset.indexed_by(**slicers)
    local_dataset = subset_time(local_dataset, query['hours'])
    return local_dataset


def forecast(query):
    forecast_fetchers = {'gridded': gridded_forecast,
                         'spot': spot_forecast,}
    return forecast_fetchers[query['type']](query)


def spot_forecast(query):
    modified_query = query.copy()
    modified_query['vars'] = ['wind', 'press']
    lat = query['location']['latitude']
    lon = query['location']['longitude']
    modified_query['domain'] = {'N': lat + 0.5,
                                'S': lat - 0.5,
                                'E': np.mod(lon + 180.5, 360.) - 180.,
                                'W': np.mod(lon + 179.5, 360.) - 180.}

    fcst = gridded_forecast(modified_query)

    def bilinear_weights(grid, x):
        # take the two closest points
        assert grid.ndim == 1
        weights = np.zeros(grid.size)
        dists = np.abs(grid - x)
        inds = np.argsort(dists)[:2]
        weights[inds] = 1. - dists[inds] / np.sum(dists[inds])
        return weights

    lat_weights = bilinear_weights(fcst[conv.LAT].data, lat)
    lon_weights = bilinear_weights(fcst[conv.LON].data, lon)
    weights = reduce(np.multiply, np.meshgrid(lat_weights, lon_weights))
    weights /= np.sum(weights)

    spot = fcst.indexed_by(**{conv.LAT: [0]})
    spot = spot.indexed_by(**{conv.LON: [0]})
    spatial_variables = [k for k, v in spot.variables.iteritems()
                            if (conv.LAT in v.dimensions and
                                conv.LON in v.dimensions)]
    for k in spatial_variables:
        assert conv.LAT in fcst[k].dimensions[-2:]
        assert conv.LON in fcst[k].dimensions[-2:]
        interpolated = np.sum(np.sum(fcst[k].data * weights.T, axis=-1), axis=-1)
        spot[k].data[:] = interpolated.reshape(spot[k].data.shape)
    # The assignments below have no effect; lat/lon returned with incorrect 0.
    # element (see spot assignments above).
    spot[conv.LAT].data[:] = lat
    spot[conv.LON].data[:] = lon
    return spot


def opendap_forecast(model):
    """
    A convenience wrapper which will looked up the uri
    to the latest openDAP dataset for source.
    """
    for server in _servers:
        try:
            latest_opendap = latest(model, server)
            logger.debug(latest_opendap)
            return xray.open_dataset(latest_opendap)
        except Exception, e:
            logger.warn("Attempt to fetch %s on %s failed."
                        % (model, server))
            logger.warn(str(e))
            pass

    for server in _servers:
        try:
            ds = fallback(model, server)
            return ds
        except Exception, e:
            logger.warn("Attempt to directly access %s files on %s failed."
                        % (model, server))
    raise ValueError("Couldn't access %s data on %s" %
                     (model, ' or '.join(_servers)))


def gridded_forecast(query):
    """

    """
    fcst = opendap_forecast(query['model'])

    def lookup_name(possible_names):
        actual_names = fcst.variables.keys()
        possible_names = [x.lower() for x in possible_names]
        name = [x for x in actual_names if x.lower() in possible_names]
        assert len(name) == 1
        return name.pop()

    variables = {}
    if 'wind' in query['vars']:
        uwnd_name = lookup_name(['u-component_of_wind_height_above_ground'])
        variables[uwnd_name] = conv.UWND
        vwnd_name = lookup_name(['v-component_of_wind_height_above_ground'])
        variables[vwnd_name] = conv.VWND
    if len(set(['rain', 'precip']).intersection(query['vars'])):
        precip_name = lookup_name(['Precipitation_rate_surface_Mixed_Intervals_Average'])
        variables[precip_name] = conv.PRECIP
    if len(set(['press', 'pressure', 'mslp']).intersection(query['vars'])):
        press_name = lookup_name(['Pressure_reduced_to_MSL_msl',
                                  'Pressure_reduced_to_MSL'])
        variables[press_name] = conv.PRESSURE
    if len(variables) == 0:
        raise ValueError("No valid variables in query")

    lat_name = lookup_name(['lat', 'latitude'])
    lon_name = lookup_name(['lon', 'longitude'])
    # reduce the datset to only the variables we care about
    fcst = fcst.select(*variables.keys())
    renames = variables.copy()
    # sometimes the time coordinate has a suffix number
    time_name = [d for d in fcst.dimensions if d.startswith('time')]
    if len(time_name) != 1:
        raise ValueError("Expected a single time dimension")
    time_name = time_name[0]
    renames.update({lat_name: conv.LAT,
                    lon_name: conv.LON,
                    time_name: conv.TIME,
                    })
    fcst = fcst.rename(renames)
    logger.debug("Selected out variables: %s" % ', '.join(variables.keys()))
    # wind speed may come at several heights so we find the 10m wind speed
    additional_slicers = {}
    dims_to_squeeze = []
    if 'wind' in query['vars']:
        height_coordinate = [d for d in fcst[conv.UWND].dimensions
                             if 'height_above_ground' in d]
        assert len(height_coordinate) == 1
        height_coordinate = height_coordinate[0]
        ind = np.nonzero(fcst[height_coordinate].data[:] == 10.)[0][0]
        additional_slicers[height_coordinate] = slice(ind, ind + 1)
        dims_to_squeeze.append(height_coordinate)
    # reduce the dataset to only the domain we care about
    # this step may take a while because it may require actually
    # downloading some of the data
    fcst = subset(fcst, query, additional_slicers)
    logger.debug("Subsetted to the domain")
    # Remove the height above ground dimension
    if len(dims_to_squeeze):
        fcst = fcst.squeeze(dims_to_squeeze)
    # normalize to the expected units etc ...
    return units.normalize_variables(fcst)
