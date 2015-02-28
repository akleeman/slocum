import xray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches

from slocum.lib import tinylib, units
from slocum.lib import conventions as conv


beaufort_in_knots = tinylib._beaufort_scale * 1.94384449
beaufort_colors = [
          '#ffffff',# white
          '#d7d7d7',# light grey
          '#a1eeff',# lightest blue
          '#42b1e5',# light blue
          '#60fd4b',# green
          '#1cea00',# yellow-green
          '#fbef36',# yellow
          '#fbc136',# orange
          '#ff4f02',# red
          '#d50c02',# darker-red
          '#ff00c0',# red-purple
          '#b30d8a',# dark purple
          '#000000',# black
          ]
wind_cmap = plt.cm.colors.ListedColormap(beaufort_colors, 'beaufort_map')
wind_cmap.set_over('0.25')
wind_cmap.set_under('0.75')

wind_norm = plt.cm.colors.BoundaryNorm(beaufort_in_knots, wind_cmap.N)


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


def spot_plot(fcsts):
    """
    Takes a set of SPOT forecasts and produces a probabilistic plot
    of the variables that are available.
    """
    assert fcsts[conv.LON].size == 1
    assert fcsts[conv.LAT].size == 1
    if fcsts[conv.LON].ndim == 1:
        fcsts = fcsts.isel(**{conv.LON: 0})
    if fcsts[conv.LAT].ndim == 1:
        fcsts = fcsts.isel(**{conv.LAT: 0})

    plotters = {'wind_speed': wind_spread_plot,
                'pressure': pressure_spread_plot,}
    variables = set(plotters.keys()).intersection(fcsts.keys())

    fig, axes = plt.subplots(len(variables), 1, sharex=True,
                             figsize=(fcsts['time'].size / 2.6, len(variables) * 4))

    if len(variables) == 1:
        axes = [axes]

    for v, ax in zip(variables, axes):
        plotters[v](fcsts, ax=ax)

    times = fcsts[conv.TIME].values
    times = times.astype('M8[m]')
    time_units = fcsts[conv.TIME].encoding[conv.UNITS]
    time_units = '%s UTC' % time_units.lower().replace('hours since ', '')

    lat = fcsts[conv.LAT].values
    lon = fcsts[conv.LON].values

    # the number of ensembles
    n = fcsts.dims[conv.ENSEMBLE]

    # we'll need both strings, and datetime.datetime time representations
    str_times = [x.strftime('%m-%d %Hh') for x in pd.to_datetime(times)]
    ax.xaxis.set_ticks(np.arange(len(str_times)) + 0.5)
    ax.xaxis.set_ticklabels(str_times, rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    title = ("Forecast for %.1fN %.1fE using %d forecasts initialized %s" %
             (lat, lon, n, time_units))
    fig.suptitle(title, fontsize=14)


def pressure_spread_plot(fcst, ax=None):
    press = fcst[conv.PRESSURE]
    scale = tinylib._pressure_scale
    spread_plot(press, scale, ax=ax)
    ax.set_title("Pressure (MSL)", fontstyle='oblique')


def wind_spread_plot(fcst, ax=None):
    """
    Adds wind speed with wind direction circles on the top of the plot.
    """
    if ax is None:
        plt.gca()

    wind_speed = fcst[conv.WIND_SPEED]
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

    # add a wind circle for each time
    for i, (_, one_time) in enumerate(fcst[conv.WIND_DIR].groupby(conv.TIME)):
        circle = WindCircle(i + 0.5, max_bin - 0.5,
                            np.ones(one_time.shape), one_time.values,
                            0.45, ax=ax,
                            cmap=plt.cm.get_cmap('Blues'),
                            norm=plt.Normalize(vmin=0., vmax=1),
                            wind_alpha=0.3)

    ax.set_ylabel("Wind Speed (knots)")
    ax.set_title("Wind", fontstyle='oblique')


def spread_plot(variable, bin_divs, ax=None):
    if ax is None:
        ax = plt.gca()

    assert variable.dims == (conv.TIME, conv.ENSEMBLE)
    # the number of ensembles
    n = variable.shape[variable.dims.index(conv.ENSEMBLE)]
    # we assume that the CF units reference time is the time the
    # forecast was created.
    times = variable[conv.TIME].values
    times = times.astype('M8[m]')

    bins = np.digitize(variable.values.reshape(-1), bin_divs)
    bins = bins.reshape(variable.shape)
    assert np.all(bins > 0)

    pm = square_bin(variable.values.T, bin_divs, zorder=-100, ax=ax)
    xs = np.arange(times.size) + 0.5
    ys = np.arange(bin_divs.size)
    ax.plot(xs, bins - 0.5, alpha=0.2)

    wind_speeds = np.round(bin_divs, 0).astype('int')
    ax.yaxis.set_ticks(ys)
    ax.yaxis.set_ticklabels(wind_speeds)

    ax.yaxis.grid(True, color='grey', which='major', alpha=0.1)
    ax.xaxis.grid(True, color='grey', which='major', alpha=0.1)

    ax.set_axisbelow(False)

    ax.set_xlim([0, np.max(xs) + 0.5])
    ax.set_ylim([0, np.max(bins) + 1])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="2%")
    cb = plt.colorbar(pm, cax=cax, ax=ax)
    cb.set_label("Forecast Probability")

    min_bin = np.sum(bin_divs <= np.min(variable.values)) - 2
    min_bin = np.maximum(min_bin, 0)
    max_bin = np.sum(bin_divs <= np.max(variable.values)) + 1
    max_bin = np.minimum(max_bin, variable.size)
    ax.set_ylim([min_bin, max_bin])


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
                         norm=plt.Normalize(vmin=0., vmax=1.),
                         *args, **kwdargs)


def gridded_plot_single_time(fcst):
    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap

    print fcst['time'].values
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    west_lon = np.mod(fcst[conv.LON].values[0] - 0.5, 360)
    east_lon = np.mod(fcst[conv.LON].values[-1] + 0.5, 360)

    m = Basemap(projection='cyl',
                llcrnrlat=np.min(fcst[conv.LAT].values) - 0.5,
                urcrnrlat=np.max(fcst[conv.LAT].values) + 0.5,
                llcrnrlon=west_lon,
                urcrnrlon=east_lon,
                resolution='i',
                ax=ax)

    radius = min(np.diff(fcst[conv.LAT])[0], np.diff(fcst[conv.LON])[0])

    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90., 91., 181.))
    m.drawmeridians(np.arange(-180., 181., 361.))

    for la, one_lat in fcst.groupby(conv.LAT):
        for lo, one_lonlat in one_lat.groupby(conv.LON):
            x, y = m(np.mod(lo, 360), la)
            circle = WindCircle(x, y,
                                speeds=one_lonlat['wind_speed'].values,
                                directions=one_lonlat['wind_dir'].values,
                                radius=0.4 * radius,
                                cmap=wind_cmap,
                                norm=wind_norm)
    cax = fig.add_axes([0.92, 0.05, 0.05, 0.9])
    mpl.colorbar.ColorbarBase(cax, cmap=wind_cmap, norm=wind_norm)

    def onclick(event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
            event.button, event.x, event.y, event.xdata, event.ydata)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


def gridded_plot(fcsts):
    for _, fcst in fcsts.groupby('time'):
        gridded_plot_single_time(fcst)


class WindCircle(object):

    def __init__(self, x, y, speeds, directions, radius,
                 cmap=None, norm=None, ax=None, fig=None,
                 wind_alpha=0.7, circle_alpha=0.6):
        self.axis, self.fig = axis_figure(ax, fig)
        self.x = x
        self.y = y
        self.center = np.array([self.x, self.y]).flatten()
        self.radius = radius
        self.cm = cmap
        self.norm = norm
        self.wind_alpha = wind_alpha
        self.circle_alpha = circle_alpha
        self.speeds = speeds
        self.directions = directions
        self.polys = self._build_polys(self.speeds, self.directions)
        self.circle = self._build_circle()
        self.axis.add_patch(self.circle)
        [self.axis.add_patch(poly) for poly in self.polys]

    def _build_circle(self):
        return patches.Circle([self.x, self.y], radius=self.radius,
                                edgecolor='k', facecolor='w',
                                zorder=10, alpha=self.circle_alpha)

    def radial(self, theta):
        return self.center + self.radius * np.array([np.sin(theta), np.cos(theta)])

    def _poly(self, speed, direction):
        xy = np.vstack([self.center,
                        self.radial(direction - np.pi / 16.),
                        self.radial(direction),
                        self.radial(direction + np.pi / 16.)])
        color = self.cm(self.norm(np.atleast_1d(speed)), alpha=self.wind_alpha)[0]
        return patches.Polygon(xy, closed=True, color=color, zorder=11)

    def _build_polys(self, speeds, directions):
        speeds = speeds.reshape(-1)
        inds = np.argsort(speeds)
        speeds = speeds[inds]
        directions = directions.reshape(-1)[inds]
        isvalid = np.logical_and(np.isfinite(speeds), np.isfinite(directions))
        return [self._poly(ws, wd)
                for ws, wd in zip(speeds[isvalid], directions[isvalid])]

    def update(self, speeds, directions):
        new_polys = self._build_polys(speeds, directions)
        for poly, new_poly in zip(self.polys, new_polys):
            poly.xy[:] = new_poly.xy[:]
            poly.set_facecolor(new_poly.get_facecolor())
            poly.set_edgecolor(new_poly.get_edgecolor())
        self.fig.canvas.blit(self.axis.bbox)

    def draw(self):
        self.axis.draw_artist(self.circle)
        [self.axis.draw_artist(poly) for poly in self.polys]
        self.fig.canvas.blit(self.axis.bbox)
