#!/usr/bin/python2.7

import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import cm
import xray
from sl.lib import units
from sl.lib import conventions as conv
from sl.lib.objects import NautAngle

_fcst_vars = [(conv.WIND_SPEED, 'knot'),
             (conv.PRESSURE, 'hPa')]


def plot_spot_ensemble(fcsts, f_var=None, plot_type='box', save_path='None'):
    """
    Plots the spread of values for forecast variable f_var in an ensemble
    forecast along the forecast times

    Parameters:
    -----------
    fcsts: sequence
        List dictionaries with members of the ensemble.
    plot_type: str
        Either 'bar' or 'box'.
    f_var: str
        Forecast variable to plot. If no variable is specified, a combined
        box plot for pressure and wind will be generated.
    save_path: str
        If specified, the plot will be saved in directory save_path under the
        file name 'se_<lat,lon>_<t0>_<plot_type>.svg'. If not specified, the
        plot will displayed interactively.
    """
    plot_handler = {
            'bar': _plot_bar,
            'box': _plot_box
            }

    n = len(fcsts)
    f0 = fcsts[0]
    t = f0[conv.TIME]
    t0_stamp = np.datetime64(
            t[2][conv.UNITS].replace('hours since ', '') + 'Z')
    lat = f0[conv.LAT][1][0]
    lon = f0[conv.LON][1][0]
    f_times = xray.decode_cf_datetime(t[1], t[2][conv.UNITS])
    f_times = f_times.astype('M8[h]')

    title = ("SPOT Ensemble for lat: %.1f lon: %.1f\n($n = %d$, $t_0 =$ %s)" %
            (lat, lon, n, t0_stamp.item().strftime('%Y-%m-%dT%HZ')))

    # not pretty but works for now...
    data_list = []
    for var, plot_units in _fcst_vars:
        fv_units = f0[var][2][conv.UNITS]
        data = [units.convert_array(f[var][1].ravel(), fv_units,
                plot_units) for f in fcsts]
        if var == f_var:
            plot_handler[plot_type](
                    f_var, data, plot_units, f_times, title)
            break
        else:
            data_list.append(data)
    # print combined plot with data_list if no forecast variable
    # was specified:
    if not f_var:
        fig, axes = plt.subplots(2, 1, sharex=True)
        fig.suptitle(title)
        for i, (var, plot_units) in enumerate(_fcst_vars):
            ax = axes[i]
            ax.boxplot(np.array(data_list[i]))
            x_labels = [ft.item().strftime('%d/%HZ') for ft in f_times]
            ax.set_ylabel("%s [%s]" % (var, plot_units))
            ax.grid(axis='y')
        ax.set_xticklabels(x_labels, rotation='vertical')

    if save_path:
        lat_name = 'S' if lat < 0 else 'N'
        lon_name = 'W' if lon < 0 else 'E'
        time_str = t0_stamp.item().strftime('%Y%m%dT%HZ')
        file_name = 'se_%s_%4.1f%s-%5.1f%s' % (time_str,
                abs(lat), lat_name, abs(lon), lon_name)
        if f_var:
            file_name += '_%s' % f_var
        if plot_type:
            file_name += '_%s' % plot_type
        file_name += '.svg'
        plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def _plot_bar(f_var, data, plot_units, f_times, title):

    plot_cols = int(np.ceil(np.sqrt(len(f_times))))

    df = pd.DataFrame(data, columns=f_times)

    dcount = df.apply(pd.value_counts).fillna(0)
    new_index = np.array(dcount.index).astype('f8').round().astype('i4')
    dcount.index = new_index
    xlim = (0, np.ceil(dcount.max().max()))

    plot_rows = int(np.ceil(len(f_times) / float(plot_cols)))
    fig, axes = plt.subplots(plot_rows, plot_cols, sharex=True, sharey=True)
    fig.suptitle("%s: %s" % (f_var, title))

    i, j = (0, 0)
    for count, t in enumerate(f_times):
        i = count / plot_cols
        j = count % plot_cols
        p_title = t.item().strftime('%d/%HZ')
        dcount[t].plot(kind='barh', ax=axes[i, j], xlim=xlim,
                color='k', alpha=0.7, fontsize='small')
        axes[i, j].set_title(p_title, fontsize='small')
        if j == 0:
            axes[i, j].set_yticklabels(dcount.index, fontsize='small')
            axes[i, j].set_ylabel("%s" % plot_units)
        if i == plot_rows - 1:
            axes[i, j].set_xlabel("count")
        plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)


def _plot_box(f_var, data, plot_units, f_times, title):

    fig, ax = plt.subplots()
    ax.boxplot(np.array(data))
    x_labels = [ft.item().strftime('%d/%HZ') for ft in f_times]
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.set_ylabel("%s [%s]" % (f_var, plot_units))
    ax.grid(axis='y')
    ax.set_title(title)


def make_gridded_ensemble(fcst_gfs, fcst_ens, moment, normalizer):
    """
    Calculates the average windspeed deviation of ensemble forecast members
    over the published GSF forecast for each grid point and forecast time
    (considering only positive deviations, i.e. ensemble member wind speed >
    GFS forecast windwpeed).

    Parameters:
    -----------
    fcst_gfs: xray.Dataset
        Dataset with the GFS forecast for windspeed (uwnd and vwnd; other
        forecast variables will be ignored if present). Must have the same
        coordinates and coordinate values as fcst_ens.
    fcst_ens: xray.Dataset
        Dataset with the GFS ensemble members that correspond to the uwnd and
        vwnd data in fcst_gfs. 'ens' must be the first dimension for
        fcst_ens['uwnd'] and fcst_ens['vwnd'].
    moment: float
        The exponent to be used in calculating the average deviation of
        ensemble wind speeds from GFS forecast windspeeds. See function
        ``_ens_spread_high`` for details.
    normalizer: float
        A windspeed in default units that will be used to make the ensemble
        deviations dimensionless.  Each positive deviation will be divided by
        normalizer before being raised to the *moment*th power and summed
        across all ensemble members. See function ``_ens_spread_high`` for
        details.

    Returns:
    --------
    A copy of fcst_gfs with the 'wind speed spread indicator' as an additional
    variable (conv.ENS_SPREAD_WIND_SPEED)
    """
    assert fcst_ens[conv.UWND].dimensions[0] == conv.ENSEMBLE
    assert fcst_ens[conv.VWND].dimensions[0] == conv.ENSEMBLE
    # for now we require ensemble member and gfs to have the same coordinates
    # and coordinate values (we're only testing for shape rather than values
    # though)
    # TODO: take intersection of gfs and ensemble member coordinates
    assert (fcst_ens[conv.UWND].dimensions[1:] ==
            fcst_gfs[conv.UWND].dimensions)
    assert fcst_ens[conv.UWND].shape[1:] == fcst_gfs[conv.UWND].shape
    assert fcst_ens[conv.TIME].data[0] == fcst_gfs[conv.TIME].data[0]

    # just in case and because normalizer will be in default units:
    units.normalize_variables(fcst_ens)
    units.normalize_variables(fcst_gfs)

    def wind_speed(u, v):
        return np.sqrt(np.power(u, 2.) + np.power(v, 2.))

    ws_gfs = wind_speed(fcst_gfs[conv.UWND].data, fcst_gfs[conv.VWND].data)
    ws_ens = wind_speed(fcst_ens[conv.UWND].data, fcst_ens[conv.VWND].data)
    # you gotta love numpy broadcasting...
    delta_ws = ws_ens - ws_gfs
    ws_spread = _ens_spread_high(delta_ws, moment, normalizer)

    # add spread data to copy of fcst_gfs:
    gfsx = fcst_gfs.copy()
    gfsx[conv.ENS_SPREAD_WIND_SPEED] = (
            fcst_gfs['uwnd'].dimensions, ws_spread, {'power': moment,
                'normalizer': normalizer})

    return gfsx


def _ens_spread_high(delta, moment, normalizer):
    """
    Calculates indicator for ensemble spread, considering only positive
    deviations.

    Parameter:
    ----------
    delta: array
        Array with differentces between ensemble and GFS forecast values for
        all ensemble members.  First dimension must be ensembles. Only
        positive differences (ensemble value higher than GFS) will be
        considered in calculation.
    normalizer: float
        Used to make delta values dimensionless (division by normalizer).
    moment: float
        'Moment' to be used in the calculation. Positive deltas (divided by
        normalizer) will taken to the *moment*th power, summed across all
        ensemble members, and then the *power*th root will be taken of the
        sum, before dividing by the number of ensemble members.

    Returns:
    --------
    Array with ensemble deviation indicator for each forecast time at each
    grid point. Shape is delta.shape[1:].
    """
    pos_delta_norm = np.power(
            np.where(delta > 0, delta, 0) / float(normalizer), moment)
    return (np.power(
            pos_delta_norm.sum(0), 1. / moment) / float(delta.shape[0]))


def plot_gridded_ensemble(gfsx, cols=2, save_path=None):
    """
    Plots the average windspeed deviation of ensemble forecast members over
    the published GSF forecast for each grid point and forecast time.
    The result is (for each forecast time) a color filled contour plot
    indicating the ensemble deviation superimposed over the usual wind barbs
    with the published GFS forecast vectors.

    Parameters:
    -----------
    fcst_gfsx: xray.Dataset
        Dataset with the GFS forecast for windspeed (uwnd and vwnd) and
        ensemble forecast deviation (other forecast variables will be ignored
        if present).
    cols: int
        Number of columns for subplot layout
    save_path: string
        If specified the plot will be saved into this directory with file
        name ``ens_<bounding box>_<t0>.svg`` where *bounding box* is
        specified as *ll_lat, ll_lon - ur_lat, ur_lon*.  If not specified or
        ``None`` the plot will be displayed interactively.
    """
    f_times = gfsx[conv.TIME].data
    for v in (conv.UWND, conv.VWND):
        units.convert_units(gfsx[v], 'knot').data
    lats = gfsx[conv.LAT].data
    lons = gfsx[conv.LON].data
    # Basemap.transform_vector requires lats and lons each to be in ascending
    # order:
    lat_inds = range(len(lats))
    lon_inds = range(len(lons))
    if NautAngle(lats[0]).is_north_of(NautAngle(lats[-1])):
        lat_inds = list(reversed(lat_inds))
        lats = lats[lat_inds]
    if NautAngle(lons[0]).is_east_of(NautAngle(lons[-1])):
        lon_inds = list(reversed(lon_inds))
        lons = lons[lon_inds]

    # determine layout:
    fig_width_inches = 10
    plot_aspect_ratio = 4./3.
    rows = int(np.ceil(gfsx.dimensions[conv.TIME] / float(cols)))
    fig_height_inches = (fig_width_inches / (float(cols) * plot_aspect_ratio)
                         * rows + 2)
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
    grid = AxesGrid(fig, [0.05, 0.01, 0.95, 0.99],
                    nrows_ncols=(rows, cols),
                    axes_pad=0.7,
                    cbar_mode='single',
                    cbar_pad=0.0,
                    cbar_size=0.3,
                    cbar_location='bottom',
                    share_all=True,)

    t0_str = f_times[0].astype('M8[h]').item().strftime('%Y-%m-%dT%H:%MZ')

    # figure title
    # plt.figtext(0.5,0.97,
    #        "GFS wind forecast and ensemble wind speed deviation",
    #        horizontalalignment='center',fontsize=16)

    # heatmap color scaling and levels
    # TODO: calculate levels as function of moment and normalizer
    spread_levels = np.linspace(0., 0.1, 50)

    m = Basemap(projection='merc', llcrnrlon=lons[0], llcrnrlat=lats[0],
            urcrnrlon=lons[-1], urcrnrlat=lats[-1], resolution='l')

    for t_step, t in enumerate(f_times):

        ax = grid[t_step]
        p_title = t.astype('M8[h]').item().strftime('%Y-%m-%dT%H:%MZ')
        m.drawcoastlines(ax=ax)
        m.drawparallels(lats,labels=[1,0,0,0], ax=ax)
        m.drawmeridians(lons,labels=[0,0,0,1], ax=ax)

        # ensemble spread heatmap:
        x, y = m(*np.meshgrid(lons, lats))
        data = gfsx[conv.ENS_SPREAD_WIND_SPEED]
        data = data.indexed_by(**{conv.TIME: t_step})
        data = data.indexed_by(**{conv.LAT: lat_inds})
        data = data.indexed_by(**{conv.LON: lon_inds}).data
        cs = m.contourf(x, y, data, spread_levels, ax=ax, cmap=cm.jet)

        # wind barbs:
        u = gfsx[conv.UWND].indexed_by(**{conv.TIME: t_step})
        u = u.indexed_by(**{conv.LAT: lat_inds})
        u = u.indexed_by(**{conv.LON: lon_inds}).data
        v = gfsx[conv.VWND].indexed_by(**{conv.TIME: t_step})
        v = v.indexed_by(**{conv.LAT: lat_inds})
        v = v.indexed_by(**{conv.LON: lon_inds}).data
        # transform from spherical to map projection coordinates (rotation
        # and interpolation).
        nxv = len(lons)
        nyv = len(lats)
        barb_length = 6
        udat, vdat, xv, yv = m.transform_vector(
                u, v, lons, lats, nxv, nyv, returnxy=True)
        # plot barbs.
        m.barbs(xv, yv, udat, vdat, ax=ax, length=barb_length, barbcolor='k',
                flagcolor='r', linewidth=0.5)

        ax.set_title(t.astype('M8[h]').item().strftime('%Y-%m-%dT%H:%MZ'))

    cbar = fig.colorbar(cs, cax=grid.cbar_axes[0], orientation='horizontal')

    if save_path:
        file_name = "ens_%s%s-%s%s_%s.svg" % (
                NautAngle(lats[0]).named_str(conv.LAT),
                NautAngle(lons[0]).named_str(conv.LON),
                NautAngle(lats[-1]).named_str(conv.LAT),
                NautAngle(lons[-1]).named_str(conv.LON),
                t0_str)
        plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight')
    else:
        plt.show()

    plt.close()
