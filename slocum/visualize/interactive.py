import xray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from slocum.lib import units, angles

import spot
import velocity


class InteractiveMap(object):

    def initialize_maps(self):
        self.map_fig, self.map_axis = plt.subplots(1, 1, figsize=(8, 8))

        self.time_fig, self.time_axis = plt.subplots(1, 1, figsize=(8, 5))

        self.time_ind = 0
        self.lat_ind = self.fcsts.dims['latitude'] / 2
        self.lon_ind = self.fcsts.dims['longitude'] / 2


class InteractiveVelocity(InteractiveMap):

    def __init__(self, fcsts, velocity_variable, speed_units='knot'):
        self.variable = velocity_variable
        # create cmaps and normalizers
        self.cmap = velocity.velocity_cmap
        self.norm = plt.cm.colors.BoundaryNorm(velocity_variable.speed_bins,
                                               self.cmap.N)

        fcsts = self.variable.normalize(fcsts)
        # convert the velocity variable to speed_units
        speed = fcsts[self.variable.speed_name]
        fcsts[self.variable.speed_name] = units.convert_units(speed, speed_units)
        self.speed_units = speed_units

        # remove any times where all forecasts are nans
        all_nan = np.apply_over_axes(np.all,
                                     np.isnan(speed.values),
                                     range(speed.ndim)[1:]).reshape(-1)
        fcsts = fcsts.isel(time=np.logical_not(all_nan))

        self.fcsts = fcsts
        # this creates two figures and sets them as internal attributes
        self.initialize_maps()
        # this creates the general structure of the spatial map
        self._initialize_map()
        # this creates the temporal spot plot
        self._initialize_time()

        def on_map_click(event):
            if event.xdata is not None:
                lon, lat = self.vel_field.m(event.xdata, event.ydata, inverse=True)
                lon_diff = angles.angle_diff(fcsts['longitude'].values, lon)
                self.lon_ind = np.argmin(np.abs(lon_diff))
                self.lat_ind = np.argmin(np.abs(fcsts['latitude'].values - lat))
                self.update_time()
        self.map_fig.canvas.mpl_connect('button_press_event', on_map_click)

        def on_time_click(event):
            if event.xdata is not None:
                self.time_ind = np.round(event.xdata - 0.5).astype('int')
                self.update_map()
        self.time_fig.canvas.mpl_connect('button_press_event', on_time_click)

        plt.show()

    def _initialize_map(self):
        fcst = self.fcsts.isel(time=self.time_ind)

        self.vel_field = velocity.VelocityField(fcst, self.variable,
                                                speed_units=self.speed_units,
                                                ax=self.map_axis)
        # add a circle around the location we are currently showing
        lon = fcst['longitude'].values[self.lon_ind]
        lat = fcst['latitude'].values[self.lat_ind]
        x, y = self.vel_field.m(lon, lat)
        self.loc = self.vel_field.m.scatter(x, y, marker='o',
                                            color='red', zorder=101, s=30)

    def _initialize_time(self):
        fcst = self.fcsts.isel(latitude=self.lat_ind,
                               longitude=self.lon_ind).copy(deep=True)
        # this is here to make sure the yaxis covers the full range of data
        max_speed = np.nanmax(self.fcsts[self.variable.speed_name].values)
        self.spot_velocity = spot.spot_velocity(fcst, self.variable,
                                                max_speed=max_speed,
                                                ax=self.time_axis)
        self.time_loc = self.time_axis.plot([0.5, 0.5],
                                            self.time_axis.get_ylim(),
                                            color='red')[0]

    def update_time(self):
        # reduce to the current location
        fcst = self.fcsts.isel(latitude=self.lat_ind,
                               longitude=self.lon_ind)
        # update the spot plot (and re-draws it)
        self.spot_velocity.update(fcst)

        # move the marker on the map to the new location
        lon = fcst['longitude'].values
        lat = fcst['latitude'].values
        x, y = self.vel_field.m(lon, lat)
        self.loc.set_offsets([x, y])
        # redraw the map canvas
        self.map_fig.canvas.draw()

    def update_map(self):
        one_time = self.fcsts.isel(time=self.time_ind)

        self.vel_field.update(one_time)

        # update the time location bar on the spot plot.
        self.time_loc.set_data([self.time_ind + 0.5] * 2, self.time_loc.get_data()[1])
        self.time_fig.canvas.draw()
