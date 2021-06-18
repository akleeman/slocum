import copy
import xarray as xra
import warnings
import itertools
import numpy as np

from slocum.lib import angles
from slocum.query import utils
from functools import reduce


def angle_slicer(x, low, high, delta=None, tolerance=4):
    """
    A function which returns a slicer that will extract all
    the values in x that fall between low and high with a prescribed
    grid delta.
    """
    assert x.ndim == 1
    # determine the native grid delta.
    diffs = angles.angle_diff(x)
    diffs = np.unique(np.round(diffs, tolerance))
    # assume angles are equally spaced.
    assert diffs.size == 1
    # assume angles are increasing
    native_delta = np.abs(diffs[0])
    # round to the nearest native stride but make sure we dont hit zero
    delta = delta or native_delta
    stride = max(1, int(np.round(delta / native_delta)))

    def fully_contained(y):
        """Returns true if the range low/high is contained in y"""
        return (np.any(angles.angle_diff(low, y) >= 0) and
                np.any(angles.angle_diff(high, y) <= 0))

    high_diff = angles.angle_diff(x, high)
    low_diff = angles.angle_diff(x, low)
    inside_domain = np.logical_and(high_diff <= 0, low_diff >= 0)

    # This next bit is meant to handle slicing longitudinal
    # requests which cross zero longitude.  When that happens the
    # GEFS data needs to be sliced twice and then stitched back
    # together.
    if inside_domain[0] and inside_domain[-1]:
        # The request domain spans the boundary, we'll have to
        # slice in two chunks and rejoint the results.
        assert diffs > 0
        slice_low = angle_slicer(x, low, x[-1] - delta + 0.01)
        slice_high = angle_slicer(x, x[0], high)
        return [slice_low, slice_high]

    # returns the index which minimizes x conditional on 'cond' being true
    def argmin_conditional(x, cond):
        x = x.copy()
        x[np.logical_not(cond)] = np.nan
        return np.nanargmin(x)
    # find the indices of the values that fall on or just outside the
    # lower and higher limits.
    low_ind = argmin_conditional(-low_diff, low_diff <= 0.)
    high_ind = argmin_conditional(high_diff, high_diff >= 0.)
    # if the angles in x are descending we flip the indices around
    if low_ind > high_ind:
        high_ind, low_ind = low_ind, high_ind

    possible_slicers = [slice(low_ind - i, high_ind - i + 1, stride)
                        for i in range(stride)]
    # we strive to find a slicer that yield the fewest number of point
    # that fully contains the domain.
    possible_slicers = [s for s in possible_slicers if fully_contained(x[s])]
    if not len(possible_slicers):
        raise ValueError("Couldn't find a slice that would provide contain "
                         "the desired domain")

    return possible_slicers[0]


def latitude_slicer(lats, query):
    """
    Returns a slice object that will slice out the smallest chunk of lats
    that covers the query domain defined by query['domain']['N'] and
    query['domain']['S'].

    The resulting slice will result in latitudes which descend from north
    to south, with a grid delta that is closest to the request grid delta,
    query['resolution'].
    """
    domain = query['domain']
    # make sure north is actually north of south.
    assert query['domain']['N'] >= query['domain']['S']
    # grid_delta was made obsolete
    if 'grid_delta' in query:
        warnings.warn('grid_delta in queries is obsolete, use resolution')
    lats = np.asarray(lats, dtype=np.float32)

    lat_delta = query.get('resolution', None)
    slicer = angle_slicer(lats, domain['S'], domain['N'],
                          lat_delta)
    if isinstance(slicer, list):
        raise ValueError("The latitude query must be malformed. %s"
                         % query)

    # at least one point south of the northern edge
    assert np.any(lats[slicer] <= domain['N'])
    # north of the southern edge
    assert np.any(lats[slicer] >= domain['S'])
    # north of the northern edge
    assert np.any(lats[slicer] >= domain['N'])
    # south of the southern edge
    assert np.any(lats[slicer] <= domain['S'])
    return slicer


def longitude_slicer(lons, query):
    """
    Returns a slice object that will slice out the smallest chunk of lons
    that covers the query domain defined by query['domain']['W'] and
    query['domain']['E'].

    The resulting slice will result in longitudes which increase from west
    to east, with a grid delta that is closest to the request grid delta,
    query['resolution'].
    """
    domain = query['domain']
    lons = np.asarray(lons, dtype=np.float32)
    if 'grid_delta' in query:
        warnings.warn('grid_delta in queries is obsolete, use resolution')

    lon_delta = query.get('resolution', None)
    slices = angle_slicer(lons, domain['W'], domain['E'], lon_delta)

    if isinstance(slices, list):
        sliced = np.concatenate([lons[s] for s in slices])
    else:
        sliced = lons[slices]

    # at least one point west of the eastern edge
    assert np.any(angles.angle_diff(sliced, domain['E']) <= 0.)
    # east of the eastern edge
    assert np.any(angles.angle_diff(sliced, domain['E']) >= 0.)
    # east of the western edge
    assert np.any(angles.angle_diff(sliced, domain['W']) >= 0.)
    # west of the western edge
    assert np.any(angles.angle_diff(sliced, domain['W']) <= 0.)

    return slices


def time_slicer(time_coordinate, query):
    """
    Returns a slice object that will slice out all times
    upto the largest query['hours'].  This allows the first
    slice of the time dimension to be lazy, after which
    subset_time should be used.
    """
    ref_time = time_coordinate.values[0]
    # next step is parsing out the times
    max_hours = max(query['hours'])
    assert int(max_hours) == max_hours
    max_time = ref_time + np.timedelta64(int(max_hours), 'h')
    max_ind = np.max(np.nonzero(time_coordinate.values <= max_time)[0])
    assert max_ind > 0
    return slice(0, max_ind + 1)


def subset_time(fcst, hours):
    """
    Extracts all the forecast valid times from fcst for lead
    times of 'hours'.  If used with a remote dataset such as
    an openDAP server this will download all the data.  Instead
    consider using time_slicer first.
    """
    # we are assuming that the first time is the reference time
    ref_time = fcst['time'].values[0]
    hours = np.array(hours)
    # make sure hours are all integers
    np.testing.assert_array_almost_equal(hours, hours.astype('int'))
    times = np.array([ref_time + np.timedelta64(int(x), 'h')
                      for x in hours])
    available_times = np.array([t for t in times if t in fcst['time'].values])
    if not available_times.size:
        raise utils.BadQuery("None of the requested forecast hours exist")
    if not available_times.size == times.size:
        missing_times = [x for x, t in zip(hours, times)
                         if t not in fcst['time'].values]
        warnings.warn("No forecast found for hour(s) %s"
                      % ', '.join(map(str, missing_times)))
    return fcst.sel(time=available_times)


def subset_gridded_dataset(remote_dataset, query, additional_slicers=None):
    """
    Given a forecast (nc) and corners of a spatial subset
    this function returns the smallest subset of the data
    which fully contains the region
    """
    slicers = {'latitude': latitude_slicer(remote_dataset['latitude'],
                                           query),
               'longitude': longitude_slicer(remote_dataset['longitude'],
                                             query),
               'time': time_slicer(remote_dataset['time'], query)}
    if not additional_slicers is None:
        slicers.update(additional_slicers)

    variables = [utils.get_variable(x) for x in query['variables']]
    required_variables = list(itertools.chain(*[v.required_variables()
                                                for v in variables]))
    remote_dataset = remote_dataset[required_variables]

    def get_one_slice(one_slice):
        # Until this point all the data might live on a remote server,
        # so we'd like to download as little as possible.  As a result
        # we split the subsetting into two steps, the first can be done
        # using slicers which minimizes downloading from openDAP servers,
        # the second pulls out the actual requested domain once the data
        # has been loaded locally.
        local_dataset = remote_dataset.isel(**one_slice)
        local_dataset = subset_time(local_dataset, query['hours'])
        local_dataset.load()
        return local_dataset

    if (isinstance(slicers['longitude'], list)
        and len(slicers['longitude']) > 1):

        def modify_slice(lon_slice):
            new_slicer = copy.copy(slicers)
            new_slicer['longitude'] = lon_slice
            return new_slicer

        data_chunks = [get_one_slice(modify_slice(lon_slice))
                       for lon_slice in slicers['longitude']]
        return xra.concat(data_chunks, dim='longitude')
    else:
        return get_one_slice(slicers)


def query_containing_point(spot_query):
    modified_query = spot_query.copy()
    lat = spot_query['location']['latitude']
    lon = spot_query['location']['longitude']
    modified_query['domain'] = {'N': lat,
                                'S': lat,
                                'E': lon,
                                'W': lon}
    return modified_query


def gridded_to_point_forecast(fcst, lon, lat):
    """
    Takes a forecast and interpolates it to a single point.
    """
    if (not np.any(lon >= fcst['longitude'].values) or
        not np.any(lon <= fcst['longitude'].values)):
        raise ValueError("Longitude %6.2f not in (%6.2f, %6.2f)"
                         % (lon,
                            fcst['longitude'].min(),
                            fcst['longitude'].max()))

    if (not np.any(lat >= fcst['latitude'].values) or
        not np.any(lat <= fcst['latitude'].values)):
        raise ValueError("Latitude %6.2f not in (%6.2f, %6.2f)"
                         % (lat,
                            fcst['latitude'].min(),
                            fcst['latitude'].max()))

    def bilinear_weights(grid, x):
        # take the two closest points
        assert grid.ndim == 1
        weights = np.zeros(grid.size)
        dists = np.abs(grid - x)
        inds = np.argsort(dists)[:2]
        # if the grid is exactly where we want to interpolate
        # set the matching index to have a weight of 1.
        if np.sum(dists[inds]) == 0. and inds.size == 1:
            weights[inds] = 1.
        else:
            weights[inds] = 1. - dists[inds] / np.sum(dists[inds])
        return weights

    lat_weights = bilinear_weights(fcst['latitude'].values, lat)
    lon_weights = bilinear_weights(fcst['longitude'].values, lon)
    weights = reduce(np.multiply, np.meshgrid(lat_weights, lon_weights))
    weights /= np.sum(weights)

    spot = fcst.isel(**{'latitude': [0]})
    spot = spot.isel(**{'longitude': [0]})
    spatial_variables = [k for k, v in spot.items()
                            if ('latitude' in v.dims and
                                'longitude' in v.dims)]
    for k in spatial_variables:
        # we assumed that latitude and longitude are the last two
        # dimensions in the forecast.
        assert 'latitude' in fcst[k].dims[-2:]
        assert 'longitude' in fcst[k].dims[-2:]
        interpolated = np.sum(np.sum(fcst[k].values * weights.T, axis=-1),
                              axis=-1)
        spot[k].values[:] = interpolated.reshape(spot[k].values.shape)
    spot['latitude'] = ('latitude', [lat], spot['latitude'].attrs)
    spot['longitude'] = ('longitude', [lon], spot['longitude'].attrs)
    return spot


def subset_spot_dataset(remote_dataset, query, additional_slicers=None):
    lat = query['location']['latitude']
    lon = query['location']['longitude']
    modified_query = query_containing_point(query)
    fcst = subset_gridded_dataset(remote_dataset,
                                  modified_query,
                                  additional_slicers)
    return gridded_to_point_forecast(fcst, lon, lat)


def subset_dataset(remote_dataset, query, additional_slicers=None):
    if query.get('type', None) == 'spot':
        return subset_spot_dataset(remote_dataset, query, additional_slicers)
    else:
        return subset_gridded_dataset(remote_dataset, query, additional_slicers)
