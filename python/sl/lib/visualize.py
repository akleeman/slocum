import xray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sl.lib import tinylib, units
from sl.lib import conventions as conv


def spot_plot(fcsts):

    plotters = {'wind_speed': wind_spread_plot,
                'pressure': pressure_spread_plot,}
#                           'wind_dir': wind_dir_spread_plot}
    variables = set(plotters.keys()).intersection(fcsts.variables.keys())

    fig, axes = plt.subplots(len(variables), 1, sharex=True,
                             figsize=(len(variables) * 5, 8))
    if len(variables) == 1:
        axes = [axes]

    for v, ax in zip(variables, axes):
        plotters[v](fcsts[v], ax=ax)

    times = fcsts[conv.TIME].values
    times = times.astype('M8[m]')
    time_units = fcsts[conv.TIME].encoding[conv.UNITS]
    time_units = '%s UTC' % time_units.lower().replace('hours since ', '')

    lat = fcsts[conv.LAT].values
    lon = fcsts[conv.LON].values

    # the number of ensembles
    n = fcsts.dims[conv.ENSEMBLE]
    title = ("Lon: %.1f Lat: %.1f $n = %d$, ref_time: %s)" %
             (lat, lon, n, time_units))
    fig.suptitle(title)

    # we'll need both strings, and datetime.datetime time representations
    str_times = [x.strftime('%m-%d %Hh') for x in pd.to_datetime(times)]
    ax.xaxis.set_ticks(np.arange(len(str_times)))
    ax.xaxis.set_ticklabels(str_times, rotation=90)

    plt.tight_layout()
    plt.show()


def wind_dir_spread_plot(wind_dir, ax=None):
    from sl.lib import plotlib
    if ax is None:
        ax = plt.gca()
    xs = np.arange(wind_dir['time'].size)
    for i, dirs in enumerate(wind_dir.values.T):
        circle = plotlib.WindCircle(i + 0.5, 0., np.ones(dirs.shape), dirs, 0.3, ax=ax)
    ax.set_xlim([0, wind_dir['time'].size])
    ax.set_ylim([-0.3, 0.3])


def pressure_spread_plot(press, ax=None):
    scale = tinylib._pressure_scale
    spread_plot(press, scale, ax=ax)

    min_bin = np.sum(scale <= np.min(press.values)) - 2
    min_bin = np.maximum(min_bin, 0)
    max_bin = np.sum(scale <= np.max(press.values)) + 1
    max_bin = np.minimum(max_bin, press.size)
    ax.set_ylim([min_bin, max_bin])
    ax.set_title("Pressure (MSL)")


def wind_spread_plot(wind_speed, ax=None):
    if ax is None:
        plt.gca()

    beaufort = xray.Variable('beaufort',
                             tinylib._beaufort_scale.astype('float32'),
                             {'units': 'm/s'})
    _, beaufort_knots, _ = units.convert_units(beaufort, 'knot')
    beaufort_knots = np.round(beaufort_knots)
    units.convert_units(wind_speed, 'knot')

    spread_plot(wind_speed, beaufort_knots, ax)

    force_nums = np.arange(beaufort_knots.size - 1)
    forces = ['F-{0:<12}'.format(i) for i in force_nums]
    max_bin = np.sum(beaufort_knots <= np.max(wind_speed.values)) + 1
    max_bin = np.minimum(max_bin, beaufort_knots.size)
    ax.yaxis.set_ticks(force_nums + 0.5, minor=True)
    ax.yaxis.set_ticklabels(forces, minor=True)
    ax.set_ylim([0, max_bin])
    ax.set_ylabel("Wind Speed (knots)")
    ax.set_title("Wind Speed")


def spread_plot(variable, bin_divs, ax=None):
    if ax is None:
        ax = plt.gca()

    assert variable.dims == (conv.ENSEMBLE, conv.TIME)
    # the number of ensembles
    n = variable.shape[variable.dims.index(conv.ENSEMBLE)]
    # we assume that the CF units reference time is the time the
    # forecast was created.
    times = variable[conv.TIME].values
    times = times.astype('M8[m]')

    bins = np.digitize(variable.values.reshape(-1), bin_divs)
    bins = bins.reshape(variable.shape)
    assert np.all(bins > 0)

    pm = square_bin(variable.values, bin_divs, zorder=-100, ax=ax)
    xs = np.arange(times.size) + 0.5
    ys = np.arange(bin_divs.size)
    ax.plot(xs, bins.T - 0.5, alpha=0.2)

    wind_speeds = np.round(bin_divs, 0).astype('int')
    ax.yaxis.set_ticks(ys)
    ax.yaxis.set_ticklabels(wind_speeds)

    ax.yaxis.grid(True, color='grey', which='major', alpha=0.1)
    ax.xaxis.grid(True, color='grey', which='major', alpha=0.1)

    ax.set_axisbelow(False)

    ax.set_xlim([0, np.max(xs)])
    ax.set_ylim([0, np.max(bins) + 1])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="2%")
    cb = plt.colorbar(pm, cax=cax, ax=ax)
    cb.set_label("Forecast Probability")


def square_bin(y, y_bins, x=None, ax=None, *args, **kwdargs):
    if ax is None:
        ax = plt.gca()
    xbins = y.shape[1]
    n = y.shape[0]
    xs = np.arange(xbins).repeat(n)
    counts, xbins, y_bins = np.histogram2d(xs, y.T.reshape(-1),
                                          bins=(np.arange(xbins + 1), y_bins))
    probs = counts.T / np.sum(counts, axis=1)
    return ax.pcolormesh(probs, cmap=plt.cm.get_cmap('Blues'),
                         norm=plt.normalize(vmin=0., vmax=1.),
                         *args, **kwdargs)

#     if save_path:
#         lat_name = 'S' if lat < 0 else 'N'
#         lon_name = 'W' if lon < 0 else 'E'
#         time_str = t0_stamp.item().strftime('%Y%m%dT%HZ')
#         file_name = 'se_%s_%4.1f%s-%5.1f%s' % (time_str,
#                 abs(lat), lat_name, abs(lon), lon_name)
#         if f_var:
#             file_name += '_%s' % f_var
#         if plot_type:
#             file_name += '_%s' % plot_type
#         file_name += '.svg'
#         plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight')
#     else:
#         plt.show()
#
#     plt.close()
#
#
#
#
#
#
#
#
# def plot_ensemble_spot(fcsts):
#     assert fcsts.dims['latitude'] == 1
#     assert fcsts.dims['longitude'] == 1
#     fcsts = fcsts.isel(latitude=0, longitude=0)
#     spread_plot(fcsts['wind_speed'])


