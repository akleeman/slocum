import xray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import velocity

from slocum.lib import units


def bin_matrix(y, bins):
    """
    Given an ndarray (y) and a set of bins, this returns a new
    array the shape of y that indicates which bin each value fell in.
    This is done over the last dimension in y.
    """
    z = np.digitize(y.reshape(-1), bins)
    z = z.reshape(y.shape)
    assert np.all(z > 0)
    return z


def bin_probs(variable, bins):
    """
    Takes a variable that is assumed to have a realization dimension,
    then returns a new DataArray named probability in which realization
    has been converted into binned probability.
    """
    assert 'realization' in variable.dims
    assert variable.ndim == 2
    other_dim = set(variable.dims).symmetric_difference(['realization']).pop()

    variable = variable.transpose(other_dim, 'realization')
    n_other, n_realization = variable.shape

    xs = np.arange(n_other).repeat(n_realization)
    counts, xbins, bins = np.histogram2d(xs, variable.values.reshape(-1),
                                         bins=(np.arange(n_other + 1),
                                               bins))
    probs = counts.T / np.sum(counts, axis=1)
    out = xray.DataArray(probs.T,
                         coords=[variable[other_dim],
                                 xray.Coordinate('bin',
                                                 np.arange(bins.size - 1))],
                         name='probability')
    return out


def binned_probability_plot(variable, bin_divs, ax=None, **kwdargs):
    """
    Creates a plot showing the binned probability of the data
    in variable.
    """
    ax, _ = utils.axis_figure(axis=ax)
    if variable.dims == ('time', ):
        variable = (xray.concat([variable], 'realization')
                    .transpose('time', 'realization'))
    assert variable.dims == ('time', 'realization')
    n_times, n_real = variable.shape
    # compute the binned probabilities
    probs = bin_probs(variable, bin_divs)
    # default to a blue colormap placed in the background
    kwdargs['cmap'] = kwdargs.get('cmap', plt.cm.get_cmap('Blues'))
    kwdargs['zorder'] = kwdargs.get('zorder', -100)
    # plot the probabilities
    y, x = np.meshgrid(np.arange(bin_divs.size),
                       np.arange(variable['time'].size + 1))
    pm = ax.pcolormesh(x, y, probs.values,
                       norm=plt.Normalize(vmin=0., vmax=1.),
                       **kwdargs)
    return pm


def binned_line_plot(variable, bin_divs, ax=None, **kwdargs):
    """
    Creates a plot showing the binned probability of the data
    in variable.
    """
    ax, _ = utils.axis_figure(axis=ax)
    assert variable.dims == ('time', 'realization')
    n_times, n_real = variable.shape

    bins = bin_matrix(variable.values, bin_divs)
    xs = np.arange(n_times).repeat(n_real).reshape(bins.shape)
    lines = plt.plot(xs + 0.5, bins - 0.5, **kwdargs)
    return lines


def binned_plot(variable, bin_divs, ax=None, **kwdargs):
    """
    Creates a plot showing the binned probability of the data
    in variable.
    """
    pm = binned_probability_plot(variable, bin_divs, ax)
    lines = binned_line_plot(variable, bin_divs, ax, alpha=0.3)

    # out plot is linear in the bins, but the bin values may not
    # be linear, here we make sure to label so that is obvious.
    ax.yaxis.set_ticks(np.arange(bin_divs.size))
    ax.yaxis.set_ticklabels(['{0:<2}'.format(x) for x in bin_divs])
    # add a soft grid showing the bin dividers
    ax.yaxis.grid(True, color='grey', which='major', alpha=0.1)
    ax.xaxis.grid(True, color='grey', which='major', alpha=0.1)
    # Turn off the lower axis
    ax.set_axisbelow(False)
    # crop the bin axis to show only the largest bin plus a buffer
    min_bin = np.sum(bin_divs <= np.min(variable.values)) - 2
    min_bin = np.maximum(min_bin, 0)
    max_bin = np.sum(bin_divs <= np.max(variable.values)) + 1
    max_bin = np.minimum(max_bin, variable.size)
    ax.set_ylim([min_bin, max_bin])

    # Add a colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="2%")
    cb = plt.colorbar(pm, cax=cax, ax=ax)
    # if it appears we actually have a set of ensemble
    # forecasts we add a colorbar indicating probability,
    # otherwise add a warning.
    cb.set_label("Forecast Agreement")
    return pm, lines


def template_temporal_plot(fcst, ax=None, fig=None):
    """
    Adds a human readable time axis to a figure.
    """
    ax, fig = utils.axis_figure(ax, fig)
    # set the x-axis to show time
    str_times = [x.strftime('%m-%d %Hh')
                 for x in pd.to_datetime(fcst['time'].values)]
    ax.xaxis.set_ticks(np.arange(len(str_times)) + 0.5)
    ax.xaxis.set_ticklabels(str_times, rotation=90)
    ax.set_xlabel("Forecast Time (UTC)")
    return ax


def ensure_single_location(fcsts):
    if 'latitude' in fcsts.coords or 'longitude' in fcsts.coords:
        # make sure the forecast is for a single location
        if (not fcsts['longitude'].size == 1 or
            not fcsts['latitude'].size == 1):
            raise ValueError("Spot forecasts required a single "
                             "latitude and longitude")
        # reduce out longitude and latitude
        if fcsts['longitude'].ndim == 1:
            fcsts = fcsts.isel(longitude=0)
        if fcsts['latitude'].ndim == 1:
            fcsts = fcsts.isel(latitude=0)
    return fcsts


class SpotVelocityPlot(object):

    def __init__(self, fcsts, velocity_variable, speed_units='knot',
                 max_speed=None, ax=None, fig=None):
        self.variable = velocity_variable
        self.speed_units = speed_units
        self.ax, self.fig = utils.axis_figure(ax, fig)
        self.ax.set_ylabel("Speed (%s)" % speed_units)

        # convert the bins to knots
        bins = xray.Variable('bins',
                             velocity_variable.speed_bins.copy(),
                             {'units': velocity_variable.units})
        _, self.bins, _ = units.convert_units(bins, speed_units)

        # determine the upper bound on speed so we can place
        # the direction circles.
        all_speeds =fcsts[self.variable.speed_name].values
        max_speed = max_speed or np.max(all_speeds)
        max_bin = np.sum(self.bins <= max_speed) + 1
        self.max_bin = np.minimum(max_bin, self.bins.size)

        # convert the data to knots and radians
        fcsts = self.normalize(fcsts)
        self.plot(fcsts)

    def normalize(self, fcsts):
        fcsts = ensure_single_location(fcsts)
        fcsts = self.variable.normalize(fcsts)

        # add a realization dimension if it doesn't exist.  This
        # allows us to support both ensemble and non-ensemble
        # forecasts with the same code.
        if not 'realization' in fcsts:
            def add_ensemble(vn):
                fcsts[vn] = (xray.concat([fcsts[vn]], 'realization')
                             .transpose('time', 'realization'))
            add_ensemble(self.variable.speed_name)
            add_ensemble(self.variable.direction_name)

        units.convert_units(fcsts[self.variable.speed_name],
                            self.speed_units)
        units.convert_units(fcsts[self.variable.direction_name],
                            'radians')
        return fcsts

    def plot(self, fcsts):
        # create the probability spread plots
        speeds = fcsts[self.variable.speed_name]
        self.prob_mesh, self.lines = binned_plot(speeds, self.bins, self.ax)
        # add a wind circle for each time
        self.circles = [self.create_circle(i, one_time)
                        for i, (_, one_time)
                        in enumerate(fcsts.groupby('time'))]
        if self.variable.speed_bin_names is not None:
            self.ax.yaxis.set_ticks(np.arange(self.bins.size - 1) + 0.5,
                                    minor=True)
            self.ax.yaxis.set_ticklabels(self.variable.speed_bin_names,
                                         minor=True)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        self.ax.set_ylim([0, self.max_bin])
        self.set_title(fcsts)

    def set_title(self, fcsts):
        title = ("Forecast for %.1fN %.1fE %s" %
                 (fcsts['latitude'].values,
                  fcsts['longitude'].values,
                  ('using %d forecasts' % fcsts.dims['realization']
                   if 'realization' in fcsts
                   else '')
                  )
                 )
        self.fig.suptitle(title, fontsize=14)

    def update(self, fcsts):
        fcsts = self.normalize(fcsts)

        # update the probabilities
        speeds = fcsts[self.variable.speed_name]
        probs = bin_probs(speeds, self.bins)
        self.prob_mesh.set_array(probs.values.ravel())

        # update the lines
        binned = bin_matrix(speeds.values, self.bins)
        # we shift down 0.5 to place lines in the center of the bins
        [x.set_data(x.get_data()[0], b - 0.5)
         for x, b in zip(self.lines, binned.T)]

        # update the directions
        iter_time = fcsts[self.variable.direction_name].groupby('time')
        for circ, (_, one_time) in zip(self.circles, iter_time):
            circ.update(np.ones(one_time.shape), one_time.values)

        # update the title
        self.set_title(fcsts)
        self.fig.canvas.draw()


    def create_circle(self, i, one_time, arrow_alpha=0.3):
        speeds = one_time[self.variable.speed_name].values
        dirs = one_time[self.variable.direction_name].values
        orientation = self.variable.direction_orientation
        circle = velocity.DirectionCircle(i + 0.5, self.max_bin - 0.5,
                                          speeds=speeds,
                                          directions=dirs,
                                          orientation=orientation,
                                          radius=0.45, ax=self.ax,
                                          cmap=plt.cm.get_cmap('Blues'),
                                          norm=plt.Normalize(vmin=-1, vmax=0),
                                          arrow_alpha=arrow_alpha)
        return circle


def spot_velocity(fcst, variable, speed_units='knot',
                  max_speed=None,
                  ax=None, fig=None):
    """
    Creates and returns a spot plot of velocity field plot.
    """
    ax = template_temporal_plot(fcst, ax, fig)
    return SpotVelocityPlot(fcst, variable, speed_units,
                            max_speed=max_speed,
                            ax=ax, fig=ax.figure)


def example_spot_single_velocity():
    import slocum.query.utils as query_utils
    n = 8
    fcst = xray.Dataset()
    times = pd.date_range('2015-01-03 06:00:00Z', periods=n, freq='6h').values
    fcst['time'] = ('time', times)
    fcst['latitude'] = ('latitude', [-8])
    fcst['longitude'] = ('longitude', [115])
    fcst['wind_speed'] = (('time', 'longitude', 'latitude'),
                          np.arange(n).reshape((n, 1, 1)),
                          {'units': 'knots'})
    fcst['wind_from_direction'] = (('time', 'longitude', 'latitude'),
                              np.linspace(-np.pi, np.pi, n).reshape((n, 1, 1)),
                              {'units': 'radians'})

    spot_plot = spot_velocity(fcst, query_utils.get_variable('wind'))
    plt.show()


def example_spot_ensemble_velocity():
    import slocum.query.utils as query_utils
    n = 17
    k = 3
    fcst = xray.Dataset()
    times = pd.date_range('2015-01-03 06:00:00Z', periods=n, freq='6h').values
    fcst['time'] = ('time', times)
    fcst['latitude'] = ('latitude', [-8])
    fcst['longitude'] = ('longitude', [115])
    fcst['wind_speed'] = (('time', 'realization', 'longitude', 'latitude'),
                          np.zeros((n, k, 1, 1)),
                          {'units': 'knots'})
    fcst['wind_from_direction'] = (('time', 'realization', 'longitude', 'latitude'),
                              np.zeros((n, k, 1, 1)),
                              {'units': 'radians'})

    fcst['wind_speed'].values[:, 0, 0, 0] = np.arange(n)
    fcst['wind_speed'].values[:, 1, 0, 0] = n - 1 - np.arange(n)
    fcst['wind_speed'].values[:, 2, 0, 0] = np.ones(n)

    fcst['wind_from_direction'].values[:, 0, 0, 0] = np.linspace(0., 2 * np.pi, n)
    fcst['wind_from_direction'].values[:, 1, 0, 0] = np.linspace(2 * np.pi, 0, n)
    fcst['wind_from_direction'].values[:, 2, 0, 0] = np.pi * np.ones(n)

    spot_plot = spot_velocity(fcst, query_utils.get_variable('wind'))
    plt.show()

if __name__ == "__main__":
    example_spot_ensemble_velocity()
    example_spot_single_velocity()
