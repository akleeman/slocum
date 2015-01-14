import xray
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import AxesGrid

from slocum.lib import units
from slocum.lib import conventions as conv
from slocum.lib.objects import NautAngle

_fcst_vars = [(conv.WIND_SPEED, 'knot'),
             (conv.PRESSURE, 'hPa')]


def plot_spot_ensemble(fcsts, f_var=None, plot_type='box', save_path=None):
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
    f_times = xray.conventions.decode_cf_datetime(t[1], t[2][conv.UNITS])
    f_times = f_times.astype('M8[h]')

    title = ("SPOT Ensemble for lat: %.1f lon: %.1f\n($n = %d$, $t_0 =$ %s)" %
            (lat, lon, n, t0_stamp.item().strftime('%Y-%m-%dT%HZ')))

    # not pretty but works for now...
    data_list = []
    for var, plot_units in _fcst_vars:
        fv_units = f0[var][2][conv.UNITS]
        data = [units.convert_array(f[var][1].ravel().copy(), fv_units,
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
    _, ax = plt.subplots()
    ax.boxplot(np.array(data))
    x_labels = [ft.item().strftime('%d/%HZ') for ft in f_times]
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.set_ylabel("%s [%s]" % (f_var, plot_units))
    ax.grid(axis='y')
    ax.set_title(title)


def make_gridded_ensemble(fcst_gfs, fcst_ens):
    """
    Calculates the average windspeed deviation of ensemble forecast members
    over the published GFS forecast for each grid point and forecast time
    (considering only positive deviations, i.e. ensemble member wind speed >
    GFS forecast windwpeed). The spread indicator will be calculated as the
    mean of the top 2 ensemble wind speed deviations vis-a-vis the GFS
    forecast.

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

    Returns:
    --------
    A copy of fcst_gfs with the 'wind speed spread indicator' as an additional
    variable (conv.ENS_SPREAD_WS)
    """
    # meta contains information about functions that can be called to calculate
    # the ensemble spread. 1st tuple element is function name, 2nd element is a
    # 'long name' that will added to the variable's attributes, and 3rd element
    # indicates the units which will be returned by the function
    # (None=dimensionless, 'default': same as underlying forecast variables;
    # alternatively a string with a unit that will be understood by
    # slocum.lib.units).
    # NOTE: attributes for unpacked fcst are currently defined in a top-level
    # dictionary in tinylib
    meta = {'topn': (_top_n_mean, 'Mean of top n (ens - gfs) deltas',
                     'default')
           }

    assert fcst_ens[conv.UWND].dimensions[0] == conv.ENSEMBLE
    assert fcst_ens[conv.VWND].dimensions[0] == conv.ENSEMBLE
    # for now we require ensemble member and gfs to have the same coordinates
    # and coordinate values (we're only testing for shape rather than values
    # though)
    # TODO: take intersection of gfs and ensemble member coordinates
    assert (fcst_ens[conv.UWND].dimensions[1:] ==
            fcst_gfs[conv.UWND].dimensions)
    assert fcst_ens[conv.UWND].shape[1:] == fcst_gfs[conv.UWND].shape
    assert fcst_ens[conv.TIME].values[0] == fcst_gfs[conv.TIME].values[0]

    # ensemble and gfs may not have same units;
    # also normalizer will be in default units:
    units.normalize_variables(fcst_ens)
    units.normalize_variables(fcst_gfs)

    def wind_speed(u, v):
        return np.sqrt(np.power(u, 2.) + np.power(v, 2.))

    ws_gfs = wind_speed(fcst_gfs[conv.UWND].values, fcst_gfs[conv.VWND].values)
    ws_ens = wind_speed(fcst_ens[conv.UWND].values, fcst_ens[conv.VWND].values)
    # you gotta love numpy broadcasting...
    delta_ws = ws_ens - ws_gfs

    spread_func = 'topn'
    ws_spread, attr = meta[spread_func][0](delta_ws, n=2)

    # add spread data to copy of fcst_gfs:
    gfsx = fcst_gfs.copy()
    attr[u'long_name'] = meta[spread_func][1]
    if meta[spread_func][2]:    # units specified
        if meta[spread_func][2] == 'default':
            attr[conv.UNITS] = fcst_gfs[conv.UWND].attrs.get(conv.UNITS)
        else:
            attr[conv.UNITS] = meta[spread_func][2]

    gfsx[conv.ENS_SPREAD_WS] = (
            fcst_gfs[conv.UWND].dimensions, ws_spread, attr)

    return gfsx


def _top_n_mean(delta, n=2):
    """
    Calculates max(0, mean of top n deltas) across ensemble members for each
    lat / lon / time step.

    delta: xray.DataArray
        Array with differentces between ensemble and GFS forecast values for
        all ensemble members.  First dimension must be ensembles.
    n: int
        Number of deltas to use for mean.

    Returns:
    --------
    Tuple consisting of:
        Numpy array with ensemble deviation indicator for each forecast time at
            each grid point (only positive deviations). Shape is
            delta.shape[1:].
        Dictionary with key 'top_x:', providing value of *top* used in
            calculation.
    """
    out_arr = np.sort(delta, axis=0)[-n:].mean(axis=0)
    attr = {'n': n}
    return np.where(out_arr > 0, out_arr, 0), attr


def plot_gridded_ensemble(gfsx, contour_units=None, max_level=None,
        barb_units='knot', cols=2, save_path=None, save_fmt='svg'):
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
    contour_units: str
        The units in which to plot the ensemble spread indicator (heatmap). If
        specified, the data sets current unit must be convertible into the
        desired unit by slocum.lib.units. If not specified the exsiting units in
        the data set will be used.
    max_level: float
        Heatmap/contour plot maximum level. Minimum level is assumed to be 0.
        Values greater tha max_level will be mapped onto the max_level color.
        If not specified, the maximum value found in the data set will be used
        as the upper end of the color bar.
    barb_units: str
        Units in which to plot the GFS wind speed/direction forecast data (wind
        bars). If specified, the data sets current unit must be convertible
        into the desired unit by slocum.lib.units.
    cols: int
        Number of columns for subplot layout.
    save_path: str
        If specified the plot will be saved into this directory with file
        name ``ens_<bounding box>_<t0>.svg`` where *bounding box* is
        specified as *ll_lat, ll_lon - ur_lat, ur_lon*.  If not specified or
        ``None`` the plot will be displayed interactively.
    save_fmt: str
        Format under which to save the image (only relevant if *save_path* is
        specified). Can be any image file extension that plt.savefig() will
        recognize as a valid format.
    """
    # adjust units as requested:
    for v in (conv.UWND, conv.VWND):
        units.convert_units(gfsx[v], barb_units)
    if contour_units:
        units.convert_units(gfsx[conv.ENS_SPREAD_WS], contour_units)
    if not max_level:
        max_level = gfsx[conv.ENS_SPREAD_WS].max()

    if isinstance(gfsx, np.datetime64): # True is gfsx has not been packed
        f_times = gfsx[conv.TIME].values
    else:                               # time variable has int offsets
        f_times = xray.conventions.decode_cf_datetime(
                gfsx['time'], gfsx['time'].attrs['units'])

    lats = gfsx[conv.LAT].values
    lons = gfsx[conv.LON].values

    # Basemap.transform_vector requires lats and lons each to be in ascending
    # order:
    lat_inds = range(len(lats))
    lon_inds = range(len(lons))
    if NautAngle(lats[0]).is_north_of(NautAngle(lats[-1])):
        lat_inds.reverse()
        lats = lats[lat_inds]
    if NautAngle(lons[0]).is_east_of(NautAngle(lons[-1])):
        lon_inds.reverse()
        lons = lons[lon_inds]

    # determine layout:
    fig_width_inches = 10
    plot_aspect_ratio = (abs(lons[-1] - lons[0]) /
                         float(abs(lats[-1] - lats[0])))
    rows = int(np.ceil(gfsx.dimensions[conv.TIME] / float(cols)))
    fig_height_inches = (fig_width_inches / (float(cols) * plot_aspect_ratio)
                         * rows + 2)
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
    fig.suptitle("GFS wind forecast in %s and "
                 "ensemble wind speed deviation" % barb_units,
                 fontsize=12)
    grid = AxesGrid(fig, [0.01, 0.01, 0.95, 0.93],
                    nrows_ncols=(rows, cols),
                    axes_pad=0.8,
                    cbar_mode='single',
                    cbar_pad=0.0,
                    cbar_size=0.2,
                    cbar_location='bottom',
                    share_all=True,)

    # size for lat/lon labels, timestamp:
    label_fontsize = 'medium' if cols <= 2 else 'small'
    # format string for colorbar labels:
    decimals = max(0, int(2 - np.floor(np.log10(max_level))))
    cb_label_fmt = '%.' + '%d' % decimals + 'f'

    # heatmap color scaling and levels
    spread_levels = np.linspace(0., max_level, 50)

    m = Basemap(projection='merc', llcrnrlon=lons[0], llcrnrlat=lats[0],
            urcrnrlon=lons[-1], urcrnrlat=lats[-1], resolution='l')

    for t_step, t in enumerate(f_times):

        ax = grid[t_step]
        m.drawcoastlines(ax=ax)
        m.drawparallels(lats,labels=[1,0,0,0], ax=ax,
                fontsize=label_fontsize)
        m.drawmeridians(lons,labels=[0,0,0,1], ax=ax,
                fontsize=label_fontsize, rotation='vertical')

        # ensemble spread heatmap:
        x, y = m(*np.meshgrid(lons, lats))
        data = gfsx[conv.ENS_SPREAD_WS]
        data = data.indexed(**{conv.TIME: t_step})
        data = data.indexed(**{conv.LAT: lat_inds})
        data = data.indexed(**{conv.LON: lon_inds}).values
        cs = m.contourf(x, y, data, spread_levels, ax=ax, extend='max',
                cmap=cm.jet)

        # wind barbs:
        u = gfsx[conv.UWND].indexed(**{conv.TIME: t_step})
        u = u.indexed(**{conv.LAT: lat_inds})
        u = u.indexed(**{conv.LON: lon_inds}).values
        v = gfsx[conv.VWND].indexed(**{conv.TIME: t_step})
        v = v.indexed(**{conv.LAT: lat_inds})
        v = v.indexed(**{conv.LON: lon_inds}).values
        # transform from spherical to map projection coordinates (rotation
        # and interpolation).
        nxv = len(lons)
        nyv = len(lats)
        barb_length = 8 - cols
        barb_width = 1.2 - (cols / 10.)
        udat, vdat, xv, yv = m.transform_vector(
                u, v, lons, lats, nxv, nyv, returnxy=True)
        # plot barbs.
        m.barbs(xv, yv, udat, vdat, ax=ax, length=barb_length, barbcolor='w',
                flagcolor='r', linewidth=barb_width)

        ax.set_title(t.astype('M8[h]').item().strftime('%Y-%m-%dT%H:%MZ'),
                fontsize=label_fontsize)

    cbar = fig.colorbar(cs, cax=grid.cbar_axes[0], orientation='horizontal',
            format=cb_label_fmt)
    attr = gfsx[conv.ENS_SPREAD_WS].attrs
    cb_label = attr.get('long_name',
            'Average (normalized) wind speed delta (ens - gfs)')
    s = ["%s = %s" % (k, attr[k]) for k in attr if k != 'long_name']
    if s:
        cb_label = "%s (%s)" % (cb_label, ', '.join(s))
    cbar.set_label(cb_label)

    if save_path:
        t0_str = f_times[0].astype('M8[h]').item().strftime('%Y%m%dT%H%MZ')
        file_name = "ens_%s%s-%s%s_%s.%s" % (
                NautAngle(lats[0]).named_str(conv.LAT),
                NautAngle(lons[0]).named_str(conv.LON),
                NautAngle(lats[-1]).named_str(conv.LAT),
                NautAngle(lons[-1]).named_str(conv.LON),
                t0_str, save_fmt)
        plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight')
    else:
        plt.show()

    plt.close()
