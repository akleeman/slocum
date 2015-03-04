import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from slocum.lib import visualize, units
from slocum.lib import conventions as conv


class InteractivePlot():

    def __init__(self, fcsts):
        self.fcsts = fcsts
        self.fcsts[conv.WIND_SPEED] = units.convert_units(self.fcsts[conv.WIND_SPEED], 'knot')

        self.map_fig, self.map_axis = plt.subplots(1, 1, figsize=(8, 8))
        self.cax = self.map_fig.add_axes([0.92, 0.05, 0.05, 0.9])

        self.time_fig, self.time_axis = plt.subplots(1, 1, figsize=(8, 5))
        self.time_ind = 0
        self.lat_ind = fcsts.dims['latitude'] / 2
        self.lon_ind = fcsts.dims['longitude'] / 2

        self._initialize_map()
        self._initialize_time()

        def on_map_click(event):
            if event.xdata is not None:
                lon, lat = self.m(event.xdata, event.ydata, inverse=True)
                self.lon_ind = np.argmin(np.abs(np.mod(fcsts['longitude'].values, 360.) - lon))
                self.lat_ind = np.argmin(np.abs(fcsts['latitude'].values - lat))
                self.update_time()
        self.map_fig.canvas.mpl_connect('button_press_event', on_map_click)

        def on_time_click(event):
            if event.xdata is not None:
                self.time_ind = np.round(event.xdata - 0.5).astype('int')
                self.update_map()
        self.time_fig.canvas.mpl_connect('button_press_event', on_time_click)

        plt.show(block=False)
        self.update_time()
        plt.show()

    def _initialize_map(self):
        print self.time_ind
        fcst = self.fcsts.isel(time=self.time_ind)
        west_lon = np.mod(fcst[conv.LON].values[0] - 0.5, 360)
        east_lon = np.mod(fcst[conv.LON].values[-1] + 0.5, 360)
        m = Basemap(projection='cyl',
                    llcrnrlat=np.min(fcst[conv.LAT].values) - 0.5,
                    urcrnrlat=np.max(fcst[conv.LAT].values) + 0.5,
                    llcrnrlon=west_lon,
                    urcrnrlon=east_lon,
                    resolution='i',
                    ax=self.map_axis)
        self.m = m

        radius = min(np.diff(fcst[conv.LAT])[0],
                     np.diff(fcst[conv.LON])[0])

        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(fcst[conv.LAT].values, labels=[1, 0, 0, 0])
        m.drawmeridians(fcst[conv.LON].values, labels=[0, 0, 0, 1], rotation=90)

        def create_circle(one_loc):
            x, y = m(np.mod(one_loc[conv.LON].values, 360),
                     one_loc[conv.LAT].values)
            return visualize.WindCircle(x, y,
                                        speeds=one_lonlat['wind_speed'].values,
                                        directions=one_lonlat['wind_dir'].values,
                                        radius=0.4 * radius,
                                        ax=self.map_axis,
                                        cmap=visualize.wind_cmap,
                                        norm=visualize.wind_norm)

        self.circles = [[create_circle(one_lonlat)
                         for lo, one_lonlat in one_lat.groupby(conv.LON)]
                        for la, one_lat in fcst.groupby(conv.LAT)]
        mpl.colorbar.ColorbarBase(self.cax,
                                  cmap=visualize.wind_cmap,
                                  norm=visualize.wind_norm)
        lon = fcst[conv.LON].values[self.lon_ind]
        lat = fcst[conv.LAT].values[self.lat_ind]
        x, y = m(np.mod(lon, 360.), lat)
        self.loc = m.scatter(x, y, marker='o', color='red', zorder=101, s=30)

        time = pd.to_datetime(fcst[conv.TIME].values)
        self.map_axis.set_title(time.strftime("Forecast for %Y-%m-%d %H:%M (UTC)"))
        return m

    def _initialize_time(self):
        fcst = self.fcsts.isel(latitude=self.lat_ind,
                               longitude=self.lon_ind).copy(deep=True)
        fcst[conv.WIND_SPEED].values[0] = np.nanmax(self.fcsts[conv.WIND_SPEED].values)
        self.spot_wind = visualize.SpotWind(fcst, ax=self.time_axis)
        self.time_loc = self.time_axis.plot([0.5, 0.5], self.time_axis.get_ylim(),
                                            color='red')[0]
        self.time_axis.set_title("Wind Speed (knots) at %4.2fN %5.2fE" %
                                 (fcst[conv.LAT], fcst[conv.LON]))

    def update_time(self):
        fcst = self.fcsts.isel(latitude=self.lat_ind,
                               longitude=self.lon_ind)
        self.spot_wind.update(fcst)
        self.time_axis.set_title("Forecast at %4.2fN %5.2fE" %
                                 (fcst[conv.LAT], fcst[conv.LON]))
        self.time_fig.canvas.draw()

        lon = fcst[conv.LON].values
        lat = fcst[conv.LAT].values
        x, y = self.m(np.mod(lon, 360.), lat)
        self.loc.set_offsets([x, y])
        self.map_fig.canvas.draw()

    def update_map(self):
        one_time = self.fcsts.isel(time=self.time_ind)
        for i, j in np.ndindex(self.fcsts.dims[conv.LAT], self.fcsts.dims[conv.LON]):
            one_lonlat = one_time.isel(latitude=i, longitude=j)
            self.circles[i][j].update(one_lonlat[conv.WIND_SPEED].values,
                                      one_lonlat[conv.WIND_DIR].values)

        self.map_fig.canvas.draw()
        time = pd.to_datetime(one_time[conv.TIME].values)
        self.map_axis.set_title(time.strftime("Forecast for %Y-%m-%d %H:%M (UTC)"))
        self.time_loc.set_data([self.time_ind + 0.5] * 2, self.time_loc.get_data()[1])
        self.time_fig.canvas.draw()