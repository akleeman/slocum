import logging
import interactive
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import basemap

from slocum.lib import angles
from slocum.query import utils
from slocum.compression import schemes


def is_spot_forecast(fcst):
    """
    Returns true if the forecast has a single latitude and longitude.
    """
    if fcst['latitude'].size == 1 and fcst['longitude'].size == 1:
        return True
    return False


def infer_variable(fcst):
    """
    This attempts to infer which variable is held in a particular forecast.
    It assumes there is a single variable.
    """
    variable = utils.available_variables(fcst)
    variable = set(variable).symmetric_difference(fcst.coords.keys())
    if len(variable) > 1:
        raise ValueError("More than one possible variable (%s) "
                         "please specify which you'd like to plot"
                         % (', '.join([v.variable_name
                                       for v in variable])))
    return variable.pop()


def axis_figure(axis=None, figure=None):
    """
    A utility function used to parse axis and figure
    arguments such that they default to the current
    figure and axis.
    """
    if not axis and not figure:
        figure = plt.gcf()
        axis = plt.gca()
    if not figure and axis:
        figure = axis.figure
    if not axis and figure:
        axis = figure.gca()
    return axis, figure


def bounding_box(fcst, pad=0.1, lon_pad=None, lat_pad=None):
    lons = fcst['longitude'].values
    lats = fcst['latitude'].values

    lon_diffs = angles.angle_diff(lons[:, None], lons)
    lat_diffs = angles.angle_diff(lats[:, None], lats)

    western_most_ind = np.nonzero(np.all(lon_diffs >= 0., axis=0))[0]
    western_most = lons[western_most_ind].item()
    eastern_most_ind = np.nonzero(np.all(lon_diffs <= 0., axis=0))[0]
    eastern_most = lons[eastern_most_ind].item()

    northern_most_ind = np.nonzero(np.all(lat_diffs <= 0., axis=0))[0]
    northern_most = lats[northern_most_ind].item()
    southern_most_ind = np.nonzero(np.all(lat_diffs >= 0., axis=0))[0]
    southern_most = lats[southern_most_ind].item()

    # count the number of lons greater than and less than each lon
    # and take the difference.  The longitude (or pair of lons) that
    # minimize this help us determine the median.  This allows different
    # definitions of longitude.
    lon_rel_loc = np.abs(np.sum(lon_diffs >= 0., axis=0) -
                         np.sum(lon_diffs <= 0., axis=0))
    central_lons = lons[lon_rel_loc == np.min(lon_rel_loc)]
    # make sure the central two aren't too far apart.
    assert np.max(central_lons) - np.min(central_lons) < 90
    median_lon = np.median(central_lons)

    lat_rel_loc = np.abs(np.sum(lat_diffs >= 0., axis=0) -
                         np.sum(lat_diffs <= 0., axis=0))
    central_lats = lats[lat_rel_loc == np.min(lat_rel_loc)]
    median_lat = np.median(central_lats)

    width = angles.geographic_distance(western_most, median_lat,
                                       eastern_most, median_lat)
    height = angles.geographic_distance(median_lon, northern_most,
                                        median_lon, southern_most)
    if lon_pad is None:
        lon_pad = pad * np.abs(angles.angle_diff(eastern_most,
                                                 western_most))
    if lat_pad is None:
        lat_pad = pad * np.abs(angles.angle_diff(northern_most,
                                                 southern_most))

    return {'llcrnrlon': western_most - lon_pad,
            'urcrnrlon': eastern_most + lon_pad,
            'urcrnrlat': northern_most + lat_pad,
            'llcrnrlat': southern_most - lat_pad,
            'width': width * (1. + 2 * pad),
            'height': height * (1. + 2 * pad),
            'lon_0': median_lon,
            'lat_0': median_lat}


def get_basemap(fcst, pad=0.1, lat_pad=None, lon_pad=None, **kwdargs):
    """
    Creates a basemap.Basemap object that bounds the forecast
    """
    kwdargs['projection'] = kwdargs.get('projection', 'cyl')
    kwdargs['resolution'] = kwdargs.get('resolution', 'i')
    bm_args = bounding_box(fcst, pad=pad, lat_pad=lat_pad, lon_pad=lon_pad)
    bm_args.update(kwdargs)
    m = basemap.Basemap(**bm_args)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents()
    m.ax.set_axis_bgcolor('#389090')
    #m.drawlsmask(land_color='#387f2b', ocean_color='#389090', grid=1.25)
    m.drawparallels(fcst['latitude'].values, labels=[1, 0, 0, 0])
    m.drawmeridians(fcst['longitude'].values, labels=[0, 0, 0, 1], rotation=90)

    return m
