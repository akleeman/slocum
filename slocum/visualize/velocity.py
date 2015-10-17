import xray
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import patches

from slocum.lib import units, angles

import utils


velocity_colors = [
          '#d7d7d7',# light grey
          '#a1eeff',# lightest blue
          '#42b1e5',# light blue
          '#4277e5',# pastel blue
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
velocity_cmap = plt.cm.colors.ListedColormap(velocity_colors, 'velocity_map')
velocity_cmap.set_over('0.25')
velocity_cmap.set_under('0.75')


class DirectionCircle(object):

    def __init__(self, x, y, speeds, directions, radius,
                 cmap=None, norm=None, ax=None, fig=None,
                 arrow_alpha=0.7, circle_alpha=0.6,
                 orientation='from'):
        self.axis, self.fig = utils.axis_figure(ax, fig)
        self.x = x
        self.y = y
        self.center = np.array([self.x, self.y]).flatten()
        self.radius = radius
        self.cm = cmap
        self.norm = norm
        self.arrow_alpha = arrow_alpha
        self.circle_alpha = circle_alpha
        assert orientation in ['from', 'to']
        self.orientation = orientation
        self.polys = self._build_polys(speeds, directions)
        self.circle = self._build_circle(speeds)
        self.axis.add_patch(self.circle)
        [self.axis.add_patch(poly) for poly in self.polys]

    def _build_circle(self, speeds):
        alpha = self.circle_alpha if np.any(np.isfinite(speeds)) else 0.
        return patches.Circle([self.x, self.y], radius=self.radius,
                                edgecolor='k', facecolor='w',
                                zorder=100, alpha=alpha)

    def radial(self, theta, radius=None):
        radius = radius or self.radius
        return self.center + radius * np.array([np.sin(theta), np.cos(theta)])

    def _poly(self, speed, direction):
        if speed > 0.:
            xy = np.vstack([self.center,
                            self.radial(direction - np.pi / 16.),
                            self.radial(direction),
                            self.radial(direction + np.pi / 16.)])
        else:
            xy = np.vstack([self.radial(ang, radius=self.radius * 0.2)
                            for ang in np.linspace(0., 2 * np.pi, 5)])
        alpha = self.arrow_alpha if np.isfinite(speed) else 0.01
        color = self.cm(self.norm(np.atleast_1d(speed)), alpha=alpha)[0]
        return patches.Polygon(xy, closed=True, color=color, zorder=101)

    def _build_polys(self, speeds, directions):
        speeds = speeds.reshape(-1)
        inds = np.argsort(speeds)
        speeds = speeds[inds]
        directions = directions.reshape(-1)[inds]
        if self.orientation == 'to':
            directions = angles.angle_add(directions, np.pi, degrees=False)
        return [self._poly(ws, wd)
                for ws, wd in zip(speeds, directions)]

    def update(self, speeds, directions):
        new_polys = self._build_polys(speeds, directions)
        for poly, new_poly in zip(self.polys, new_polys):
            poly.xy[:] = new_poly.xy[:]
            poly.set_facecolor(new_poly.get_facecolor())
            poly.set_edgecolor(new_poly.get_edgecolor())
            poly.set_alpha(new_poly.get_alpha())
        self.circle.set_alpha(self._build_circle(speeds).get_alpha())
        # no sense in blit-ing if the image hasn't been rendered yet.
        if hasattr(self.fig.canvas, 'renderer'):
            self.fig.canvas.blit(self.axis.bbox)

    def draw(self):
        self.axis.draw_artist(self.circle)
        [self.axis.draw_artist(poly) for poly in self.polys]
        self.fig.canvas.blit(self.axis.bbox)


class VelocityField(object):

    def __init__(self, fcst, velocity_variable,
                 speed_units='knots', ax=None, fig=None,
                  **kwdargs):
        self.variable = velocity_variable
        self.speed_units = speed_units
        # set the default colormap if needed
        kwdargs['cmap'] = kwdargs.get('cmap', velocity_cmap)

        # convert the bins to knots
        bins = xray.Variable('bins',
                             velocity_variable.speed_bins.copy(),
                             {'units': velocity_variable.units})
        _, self.bins, _ = units.convert_units(bins, speed_units)

        default_norm = plt.cm.colors.BoundaryNorm(self.bins, self.bins.size)
        kwdargs['norm'] = kwdargs.get('norm', default_norm)

        self.variable = velocity_variable
        self.ax, self.fig = utils.axis_figure(ax, fig)

        # add the color bar
        self.cax = self.fig.add_axes([0.92, 0.05, 0.05, 0.9])
        mpl.colorbar.ColorbarBase(self.cax,
                                  cmap=kwdargs['cmap'],
                                  norm=kwdargs['norm'])

        fcst = self.normalize(fcst)
        # use the longitude grid to define the radius of the circles
        sorted_lats = np.sort(fcst['latitude'].values)
        resol = np.median(angles.angle_diff(sorted_lats[1:],
                                            sorted_lats[:-1]))

        # create the map
        self.m = utils.get_basemap(fcst, ax=self.ax,
                                   lon_pad=0.75 * resol,
                                   lat_pad = 0.75 * resol)
        def create_circle(one_loc):
            # determine the circle center
            x, y = self.m(one_loc['longitude'].values,
                          one_loc['latitude'].values)
            speeds = one_lonlat[self.variable.speed_name].values
            dirs = one_lonlat[self.variable.direction_name].values
            orientation = self.variable.direction_orientation
            return DirectionCircle(x, y,
                                   speeds=np.atleast_1d(speeds),
                                   directions=np.atleast_1d(dirs),
                                   orientation=orientation,
                                   radius=0.4 * resol,
                                   ax=self.ax, **kwdargs)

        self.circles = [[create_circle(one_lonlat)
                         for lo, one_lonlat in one_lat.groupby('longitude')]
                        for la, one_lat in fcst.groupby('latitude')]
        self.set_title(fcst)


    def normalize(self, fcst):
        if 'time' in fcst.dims:
            if not fcst.dims['time'] == 1:
                raise ValueError("Expected a single time for VelocityField")
            fcst = fcst.isel(time=0)
        fcst = self.variable.normalize(fcst)
        units.convert_units(fcst[self.variable.speed_name],
                            self.speed_units)
        units.convert_units(fcst[self.variable.direction_name],
                            'radians')
        return fcst

    def set_title(self, fcst):
        time = pd.to_datetime(fcst['time'].values)
        self.ax.set_title(time.strftime("Forecast for %Y-%m-%d %H:%M (UTC)"))

    def update(self, fcst):
        fcst = self.normalize(fcst)
        # update each of the velocity circles
        for i, j in np.ndindex(fcst.dims['latitude'], fcst.dims['longitude']):
            one_lonlat = fcst.isel(latitude=i, longitude=j)
            dirs = one_lonlat[self.variable.direction_name].values
            speeds = one_lonlat[self.variable.speed_name].values
            self.circles[i][j].update(speeds, dirs)

        self.set_title(fcst)
        self.fig.canvas.draw()

