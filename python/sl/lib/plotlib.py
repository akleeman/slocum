import copy
import numpy as np
import coards
import logging
import itertools

import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)

try:
    from mpl_toolkits.basemap import Basemap
except:
    logging.debug("not loading basemap")

from bisect import bisect
from matplotlib import pylab, cm, colors, colorbar, patches, cbook
from matplotlib.widgets import Button, RectangleSelector

import sl.objects.conventions as conv

from sl.lib import datelib
from sl.objects import objects, core
from sl import poseidon

beaufort_colors = ['#a1eeff', # light blue
          '#42b1e5', # darker blue
          '#60fd4b', # green
          '#1cea00', # yellow-green
          '#fbef36', # yellow
          '#fbc136', # orange
          '#ff4f02', # red
          '#ff0e02', # darker-red
          '#ff00c0', # red-purple
          '#d925ac', # purple
          '#b30d8a', # dark purple
          #'#000000', # black
          ]

beaufort_cm = colors.ListedColormap(beaufort_colors, 'beaufort_map')
beaufort_cm.set_over('0.25')
beaufort_cm.set_under('0.75')
beaufort_bins = np.array([0., 1., 3., 6., 10., 16., 21., 27., 33., 40., 47., 55.])
beaufort_norm = colors.BoundaryNorm(beaufort_bins, beaufort_cm.N)

direction_colors = ['blue', 'green', 'yellow', 'orange', 'red']
direction_cm = colors.ListedColormap(direction_colors, 'wind_dir_map')
direction_bins = np.linspace(0., np.pi, 5)
direction_norm = colors.BoundaryNorm(direction_bins, direction_cm.N)

def wind_hist(arr, ax=None):
    vals, bins = np.histogram(arr, bins=beaufort_bins, normed=True)
    hist_ret = ax.fill_between(beaufort_bins[1:], vals, color='r')
    return hist_ret

def axis_figure(axis=None, figure=None):
        if not axis and not figure:
            figure = plt.gcf()
            axis = plt.gca()
        if not figure and axis:
            figure = axis.figure
        if not axis and figure:
            axis = figure.gca()
        return axis, figure

class ButtonIndex(object):
    """
    A class which keeps track of the current index
    and when clicked increments the index and calls
    drawfunc with the current ind.
    """
    def __init__(self, n, drawfunc):
        self.ind = 0
        self.n = n
        self.drawfunc = drawfunc

    def draw(self):
        i = self.ind % self.n
        self.drawfunc(i)

    def next(self, event):
        self.ind += 1
        self.draw()

    def prev(self, event):
        self.ind -= 1
        self.draw()

def frame_off(axis):
    axis.set_frame_on(False)
    axis.set_xticks([])
    axis.set_yticks([])

def plot_wind(fcst, **kwdargs):
    fig = plt.figure(figsize=(16, 8))
    # the area in which the map is ploted
    map_axis = fig.add_axes([0.25, 0.05, 0.7, 0.9])
    # the histogram axes
    info_axis = fig.add_axes([0.03, 0.15, 0.18, 0.2])
    direc_axis = fig.add_axes([0.03, 0.45, 0.18, 0.2])
    speed_axis = fig.add_axes([0.03, 0.75, 0.18, 0.2])
    # the color bar
    cb_axis = fig.add_axes([0.97, 0.05, 0.01, 0.9])
    # the previous button
    prev_ax = fig.add_axes([0.08, 0.05, 0.03, 0.04])
    # the next button
    next_ax = fig.add_axes([0.13, 0.05, 0.03, 0.04])

    info_axis.set_axis_bgcolor(fig.get_facecolor())
    frame_off(info_axis)
    # fill in the colorbar
    cb = colorbar.ColorbarBase(cb_axis, cmap=beaufort_cm,
                               norm=beaufort_norm,
                               ticks=beaufort_bins,
                               boundaries=beaufort_bins)

    original_forecast = copy.deepcopy(fcst)

    def draw_map(f, map_axis):
        lons = f[conv.LON].data
        lats = f[conv.LAT].data
        ll = objects.LatLon(min(lats), min(lons))
        ur = objects.LatLon(max(lats), max(lons))
        map = plot_map(ll, ur, ax=map_axis)
        map.fillcontinents(color='green',lake_color='white')
        map.drawparallels(lats, zorder=5)
        map.drawmeridians(lons, zorder=5)
        x, y = map(*pylab.meshgrid(lons,lats))
        map_axis.set_xticks(x[-1, :])
        map_axis.set_xticklabels(['%.0fE' % z for z in lons])
        map_axis.set_yticks(y[:, 0])
        map_axis.set_yticklabels(['%.0fN' % z for z in lats])
        return map

    m = draw_map(fcst, map_axis)

    def single_slice(fcst, index, dim):
        fc = fcst.take([index], dim)
        fc.squeeze(dim)
        return fc

    wind_map = WindMap(fcst.take([0], conv.TIME), m)

    def draw_single_time(fcst, index):
        wind_map.update(fcst.take([index], conv.TIME))
        fig.canvas.blit(map_axis.bbox)
        fig.canvas.draw()

    callback = ButtonIndex(fcst.dimensions[conv.TIME], lambda x : draw_single_time(fcst, x))
    bnext = Button(next_ax, '>')
    bnext.on_clicked(callback.next)
    bprev = Button(prev_ax, '<')
    bprev.on_clicked(callback.prev)

    wind_hist(fcst.take([callback.ind], conv.TIME)['wind_speed'].data.flatten(), speed_axis)
    speed_axis.set_title("Wind distribution")
    def show_hist(event):
        if event.inaxes:
            speed_axis.cla()
            lon, lat = m(event.xdata, event.ydata, inverse=True)
            f = fcst.take([callback.ind], conv.TIME)
            f.squeeze(conv.TIME)
            fcst_lats = np.mod(f['lat'].data + 90, 180) - 90
            requested_lat = np.mod(lat + 90, 180) - 90
            lat_ind = np.argmin(np.abs(fcst_lats - requested_lat))
            grid = f.take([lat_ind], 'lat')
            grid.squeeze('lat')
            fcst_lons = np.mod(f['lon'].data, 360)
            lon_ind = np.argmin(np.abs(fcst_lons - np.mod(lon, 360)))
            grid = grid.take([lon_ind], 'lon')
            grid.squeeze('lon')
            wind_hist(grid['wind_speed'].data.flatten(), speed_axis)
            speed_axis.set_title('Wind distribution at %d, %d' %
                                (grid['lat'].data[0], grid['lon'].data[0]))
            speed_axis.set_ylim([0, speed_axis.get_ylim()[1]])
            speed_axis.set_yticks([])
            plt.draw()
    plt.connect('button_press_event', show_hist)

    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = m(eclick.xdata, eclick.ydata, inverse=True)
        x2, y2 = m(erelease.xdata, erelease.ydata, inverse=True)
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        ll = objects.LatLon(ymin, xmin)
        ur = objects.LatLon(ymax, xmax)
        ur, ll = poseidon.ensure_corners(ll, ur)
        lon_inds = np.arange(bisect(fcst[conv.LON], ll.lon - 1.),
                             bisect(fcst[conv.LON], ur.lon + 1.))
        lat_inds = np.arange(bisect(fcst[conv.LAT], ll.lat - 1.),
                             bisect(fcst[conv.LAT], ur.lat + 1.))
        fc = fcst.take(lon_inds, conv.LON)
        fc = fc.take(lat_inds, conv.LAT)
        map_axis.cla()
        m = draw_map(fcst, map_axis)
        wind_map = WindMap(fcst.take([callback.ind], conv.TIME), m)
        plt.draw()

    def toggle_selector(event):
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            toggle_selector.RS.set_active(True)

    toggle_selector.RS = RectangleSelector(map_axis, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1,3], # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='data')

    plt.connect('key_press_event', toggle_selector)
    plt.show()
    return ax

class WindMap(object):

    def __init__(self, fcst, base_map, scale = 1):
        # This checks to make sure the time dim is length one
        fcst.squeeze(conv.TIME)
        self.base_map = base_map
        self.axis = base_map.ax
        lons = fcst['lon'].data
        lats = fcst['lat'].data
        self.x, self.y = base_map(*pylab.meshgrid(lons,lats))
        self.radius = scale * 0.45 * np.min(np.abs(np.diff(self.x)))
        self.axis.set_ylim([np.min(self.y) - self.radius, np.max(self.y) + self.radius])
        self.axis.set_xlim([np.min(self.x) - self.radius, np.max(self.x) + self.radius])
        def iter_circles():
            for lat, flat in fcst.iterator('lat'):
                flat.squeeze('lat')
                for lon, fpoint in flat.iterator('lon'):
                    fpoint.squeeze('lon')
                    x, y = base_map(lon.data, lat.data)
                    yield WindCircle(x, y,
                                     fpoint['wind_speed'].data,
                                     fpoint['wind_dir'].data,
                                     self.radius,
                                     ax=self.axis)

        fc_time = datelib.from_udvar(fcst[conv.TIME])[0]
        self.axis.set_title(fc_time.strftime('%A %B %d at %H:%M GMT'))
        if conv.PRECIP in fcst.variables:
            pp = np.sum(fcst[conv.PRECIP].data == 0.25, axis=0).astype('float')
            pp /= float(fcst[conv.PRECIP].data.shape[0])
            pp = pp.squeeze()
            self.precip_pcolor = self.axis.pcolor(self.x, self.y, pp, cmap=cm.Greens, alpha=0.5)
        self.circles = list(iter_circles())

    def update(self, fcst):
        # This checks to make sure the time dim is length one
        fcst.squeeze(conv.TIME)
        lons = fcst['lon'].data
        lats = fcst['lat'].data
        x, y = self.base_map(*pylab.meshgrid(lons,lats))
        if not np.all(self.x == x) or not np.all(self.y == y):
            raise ValueError("expected lat lons to be the same")

        def iter_point():
            for lat, flat in fcst.iterator('lat'):
                flat.squeeze('lat')
                for lon, fpoint in flat.iterator('lon'):
                    fpoint.squeeze('lon')
                    yield fpoint

        fc_time = datelib.from_udvar(fcst[conv.TIME])[0]
        self.axis.set_title(fc_time.strftime('%A %B %d at %H:%M GMT'))
        for wind_circle, point_forecast in zip(self.circles, iter_point()):
            wind_circle.update(point_forecast['wind_speed'].data,
                               point_forecast['wind_dir'].data)
        if conv.PRECIP in fcst.variables:
            pp = np.sum(fcst[conv.PRECIP].data > 0., axis=0).astype('float')
            pp /= float(fcst[conv.PRECIP].data.shape[0])
            pp = pp.squeeze()
            self.precip_pcolor.set_visible(False)
            self.precip_pcolor = self.axis.pcolor(self.x, self.y, pp, cmap=cm.Greens, alpha=0.5)

    def draw(self):
        [wind_circle.draw() for wind_circle in self.circles]

class WindCircle(object):

    def __init__(self, x, y, speeds, directions, radius,
                 cmap=None, norm=None, ax=None, fig=None):
        self.axis, self.fig = axis_figure(ax, fig)
        self.x = x
        self.y = y
        self.center = np.array([self.x, self.y]).flatten()
        self.radius = radius
        self.cm = cmap or beaufort_cm
        self.norm = norm or beaufort_norm
        self.wind_alpha = 0.7
        self.circle_alpha = 0.6
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
        speeds = speeds.flatten()
        directions = directions.flatten()
        isvalid = np.logical_and(np.isfinite(speeds), np.isfinite(directions))
        return [self._poly(ws, wd) for ws, wd in zip(speeds[isvalid], directions[isvalid])]

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

def wind(x, y, wind_speed, wind_dir, ax=None, fig=None, scale=1.):
    if not (wind_dir.shape == wind_speed.shape):
        raise ValueError("expected all arrays to be the same size")
    ax, fig = axis_figure(axis=ax, figure=fig)
    radius = scale * 0.45 * np.min(np.diff(x))

def make_pretty(m, ocean_color='#dcdcdc'):
    # map with continents drawn and filled.
    m.drawcoastlines()
    m.drawlsmask(land_color='green',ocean_color=ocean_color,lakes=True)
    m.drawcountries()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    m.drawmapboundary(fill_color='aqua')

def plot_passages_map(overlap_passages, ax=None):
    """
    Given an object with overlapping passages (ie, all STEPS of the passage
    have the same lat lon) this will plot the wind circles experienced at
    each of the steps.
    """
    lons = np.unique(overlap_passages[conv.LON].data)
    lats = np.unique(overlap_passages[conv.LAT].data)
    ll = objects.LatLon(np.min(lats), np.min(lons))
    ur = objects.LatLon(np.max(lats), np.max(lons))
    ur, ll = poseidon.ensure_corners(ur, ll)
    m = plot_map(ll, ur, ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='green',lake_color='white')
    grid_lats = np.linspace(ll.lat, ur.lat, 5, endpoint=True)
    m.drawparallels(grid_lats, zorder=5)
    grid_lons = np.linspace(ll.lon, ur.lon, 5, endpoint=True)
    m.drawmeridians(grid_lons, zorder=5)
    x, y = m(*pylab.meshgrid(grid_lons,grid_lats))
    m.ax.set_xticks(x[-1, :])
    m.ax.set_xticklabels(['%.1fE' % z for z in grid_lons])
    m.ax.set_yticks(y[:, 0])
    m.ax.set_yticklabels(['%.1fN' % z for z in grid_lats])

    waypoints = [np.array(m(lon, lat)) for lon, lat in zip(lons, lats)]
    dists = [np.linalg.norm(x - y) for x, y in itertools.combinations(waypoints, 2)]
    radius = 0.45 * np.min(dists)

    for step, conditions in overlap_passages.iterator(conv.STEP):
        lat = np.unique(conditions[conv.LAT].data)
        if lat.size > 1:
            raise ValueError("Expected a single lat")
        lon = np.unique(conditions[conv.LON].data)
        if lon.size > 1:
            raise ValueError("Expected a single lon")
        x, y = m(lon, lat)
        wind = [objects.Wind(u, v) for u, v in zip(conditions[conv.UWND].data.flatten(),
                                                   conditions[conv.VWND].data.flatten())]
        speeds = np.array([w.speed for w in wind])
        directions = np.array([w.dir for w in wind])
        circle = WindCircle(x, y, speeds, directions, radius, ax=m.ax)

    return m.ax

def plot_passages(passages, etc_var=None):
    fig = plt.figure(figsize=(16, 8))
    # the area in which the map is ploted
    map_axis = fig.add_axes([0.25, 0.05, 0.7, 0.9])
    # the histogram axes
    boat_speed_axis = fig.add_axes([0.04, 0.15, 0.18, 0.2])
    boat_speed_axis.set_title("Boat Speed")
    boat_speed_axis.set_ylim([0, 7])
    direc_axis = fig.add_axes([0.04, 0.45, 0.18, 0.2])
    direc_axis.set_title("Relative Wind Direction")
    direc_axis.set_ylim([0, np.pi])
    direc_axis.set_yticks([0, np.pi/2, np.pi])
    direc_axis.set_yticklabels(['down', 'reach', 'up'])
    wind_speed_axis = fig.add_axes([0.04, 0.75, 0.18, 0.2])
    wind_speed_axis.set_title("Wind Speed")
    # the color bar
    cb_axis = fig.add_axes([0.97, 0.05, 0.01, 0.9])
    # fill in the colorbar
    cb = colorbar.ColorbarBase(cb_axis, cmap=beaufort_cm,
                               norm=beaufort_norm,
                               ticks=beaufort_bins,
                               boundaries=beaufort_bins)

    overlap_passages = objects.intersect_ensemble_passages(passages)
    plot_passages_map(overlap_passages, ax=map_axis)

    times = datelib.from_coards(passages[conv.TIME].data.flatten(),
                                passages[conv.TIME].attributes[conv.UNITS])
    times = filter(lambda x : not x is None, times)
    units = min(times).strftime('days since %Y-%m-%d %H:00:00')
    if max(np.array([coards.to_udunits(x, units) for x in times])) <= 2:
        units = units.replace('days', 'hours')

    for _, passage in passages.iterator(conv.ENSEMBLE):
        if conv.NUM_STEPS in passage.variables:
            passage = passage.view(slice(0, passage[conv.NUM_STEPS].data), conv.STEP)
        time = datelib.from_coards(passage[conv.TIME].data.flatten(), units)
        time = np.array([coards.to_udunits(x, units) for x in time])
        boat_speed_axis.plot(time, passage[conv.SPEED].data)
        wind = [objects.Wind(u, v) for u, v in
                zip(passage[conv.UWND].data, passage[conv.VWND].data)]
        wind_speed = [x.speed for x in wind]
        wind_speed_axis.plot(time, wind_speed)
        def rel(x):
            if x < np.pi:
                return x
            else:
                return 2*np.pi - x
        rel_wind = [rel(np.abs(heading - w.dir)) for heading, w in zip(passage[conv.HEADING], wind)]
        direc_axis.plot(time, rel_wind)
    fig.suptitle('Time is in %s' % units)
    plt.show()

def plot_passage(passage):
    plot_passages([passage])

def plot_when(date_passages):

    fig = plt.figure()
    prob_axis = fig.add_axes([0.05, 0.05, 0.8, 0.4])
    beaufort_bar_axis = fig.add_axes([0.90, 0.05, 0.01, 0.4])
    direction_axis = fig.add_axes([0.05, 0.5, 0.8, 0.4])
    direction_bar_axis = fig.add_axes([0.90, 0.5, 0.01, 0.4])
    # fill in the colorbar
    colorbar.ColorbarBase(beaufort_bar_axis, cmap=beaufort_cm,
                               norm=beaufort_norm,
                               ticks=beaufort_bins,
                               boundaries=beaufort_bins)
    direction_bar = colorbar.ColorbarBase(direction_bar_axis, cmap=direction_cm,
                               norm=direction_norm,
                               ticks=direction_bins,
                               boundaries=direction_bins)
    direction_bar.set_ticklabels(['down', 'broad', 'reach', 'close', 'beat'])
    all_dates = [datelib.from_coards(x[conv.TIME].data.flatten(),
                                     x[conv.TIME].attributes[conv.UNITS]) for x in date_passages]
    start_dates = [min(filter(lambda x : not x is None, x)) for x in all_dates]
    units = min(start_dates).strftime('days since %Y-%m-%d %H:00:00')
    if max(np.array([coards.to_udunits(x, units) for x in start_dates])) <= 2:
        units = units.replace('days', 'hours')
    times = np.array([coards.to_udunits(x, units) for x in start_dates])

    # plot the percentage of wind exceedence
    def iter_probs():
        for passages in date_passages:
            speed = passages['wind_speed'].data
            valid_speed = speed[np.isfinite(speed)]
            sums = np.array([np.sum(valid_speed >= x) for x in beaufort_bins])
            yield sums / float(valid_speed.size)
    # drop the first one since wind is always > 0
    probs = np.array(list(iter_probs()))[:, 1:].T
    for date_probs, color in zip(probs, beaufort_colors):
        prob_axis.fill_between(times, date_probs, color=color)
    prob_axis.set_xlim([0, max(times)])

    # plot the precentage of direction exceedence
    def iter_direction_probs():
        for passages in date_passages:
            direction = passages['rel_wind'].data
            valid_direction = direction[np.isfinite(direction)]
            sums = np.array([np.sum(valid_direction >= x) for x in direction_bins])
            yield sums / float(valid_direction.size)

    direction_probs = np.array(list(iter_direction_probs())).T
    for dir_probs, color in zip(direction_probs, direction_colors):
        direction_axis.fill_between(times, dir_probs, color=color)
    direction_axis.set_xlim([0, max(times)])
    plt.show()
    import pdb; pdb.set_trace()

def plot_summaries(passages):

    def summarize(passage):
        ret = {}
        ret['max_wind'] = np.max(passage[conv.WIND_SPEED].data)
        ret['avg_wind'] = np.mean(passage[conv.WIND_SPEED].data)
        ret['motor_hours'] = np.sum(passage[conv.HOURS].data[passage[conv.MOTOR_ON].data == 1])
        ret['hours'] = np.sum(passage[conv.HOURS])
        upwind = passage[conv.RELATIVE_WIND].data <= np.pi / 3.
        downwind = passage[conv.RELATIVE_WIND].data >= (2. * np.pi / 3.)
        reach = np.logical_not(np.logical_or(upwind, downwind))
        ret['upwind_hours'] = np.sum(passage[conv.HOURS].data[upwind])
        ret['reach_hours'] = np.sum(passage[conv.HOURS].data[reach])
        ret['downwind_hours'] = np.sum(passage[conv.HOURS].data[downwind])
        return ret

    def start(passage):
        return datelib.from_udvar(passage['time'])[0]
    starts = [start(x) for x in passages]
    summaries = [summarize(x) for x in passages]
    summaries = dict((k, np.array([x[k] for x in summaries])) for k in summaries[0].keys())
    earliest = min(starts)
    hours = [datelib.hours(x - earliest) for x in starts]

    height = 0.75*np.min(np.diff(hours))

    upwind = summaries['upwind_hours']
    reach = summaries['reach_hours']
    downwind = summaries['downwind_hours']

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.barh(hours, upwind, height=height, color='r', align='center')
    ax1.barh(hours, reach, height=height, left=upwind, color='y', align='center')
    ax1.barh(hours, downwind, height=height, left=upwind+reach, color='g', align='center')
    ax1.set_title('Relative Wind Direction [r=up, y=reach, g=down]')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.barh(hours, summaries['motor_hours'], height=height, color='b')
    ax2.set_title('Hours motoring')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.barh(hours, summaries['max_wind'], height=height, color='r')
    ax3.set_title('Maximum Wind Speed (kts)')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.barh(hours, summaries['avg_wind'], height=height, color='g')
    ax4.set_title('Average Wind Speed(kts)')

    fig.suptitle(earliest.strftime('starting x hours after %Y-%m-%d %H:%M'))
    plt.show()

def plot_map(ll, ur, proj='lcc', pad=0.25, ax=None):
    mid = objects.LatLon(0.5*(ll.lat + ur.lat), 0.5*(ll.lon + ur.lon))

    lat_deg = max(np.abs(ll.lat - ur.lat), 2)
    lon_deg = max(np.abs(ll.lon - ur.lon), 2)

    llcrnrlon=min(ll.lon, ur.lon) - pad * lon_deg
    llcrnrlat=min(ll.lat, ur.lat) - pad * lat_deg
    urcrnrlon=max(ll.lon, ur.lon) + pad * lon_deg
    urcrnrlat=max(ll.lat, ur.lat) + pad * lat_deg

    m = Basemap(projection=proj,
                lon_0=mid.lon,
                lat_0=mid.lat,
                llcrnrlon=llcrnrlon,
                llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon,
                urcrnrlat=urcrnrlat,
                rsphere=(6378137.00,6356752.3142),
                area_thresh=1000.,
                width=np.abs(ll.lon - ur.lon),
                height=np.abs(ll.lat - ur.lat),
                resolution='l',
                ax=ax)
    make_pretty(m)
    return m

def plot_route(passage, proj='lcc'):
    lons = passage[conv.LON].data
    lats = passage[conv.LAT].data
    ll = objects.LatLon(np.min(lats), np.min(lons))
    ur = objects.LatLon(np.max(lats), np.max(lons))
    ur, ll = poseidon.ensure_corners(ur, ll)
    m = plot_map(ll, ur, proj=proj)
    xvals, yvals = m(lons,lats)
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    # draw colored markers.
    # use zorder=10 to make sure markers are drawn last.
    # (otherwise they are covered up when continents are filled)
    m.scatter(xvals, yvals, 10, edgecolors='none', zorder=10)
    m.barbs(xvals, yvals, passage[conv.UWND].data, passage[conv.VWND].data, zorder=9)
    return m

def plot_routes(routes, colors = None, proj = 'lcc'):
    first_route = routes[0]
    fig = plt.figure()

    lons = passage[conv.LON].data
    lats = passage[conv.LAT].data
    ll = objects.LatLon(np.min(lats), np.min(lons))
    ur = objects.LatLon(np.max(lats), np.max(lons))
    ur, ll = poseidon.ensure_corners(ur, ll)
    m = plot_map(ll, ur, proj=proj)

    if colors is None:
        colors = ['g'] * len(routes)
    elif not hasattr(colors, '__iter__'):
        colors = [colors] * len(routes)
    else:
        if not len(colors) == len(routes):
            raise ValueError("expected colors and routes to be same len")

    for route, color in zip(routes, colors):
        lats = route[conv.LAT].data
        lons = route[conv.LON].data
        xvals, yvals = m(lons,lats)
        xvals = np.array(xvals)
        yvals = np.array(yvals)

        # draw colored markers.
        # use zorder=10 to make sure markers are drawn last.
        # (otherwise they are covered up when continents are filled)
        m.plot(xvals, yvals, color=color, linewidth=3)
    plt.show()

def plot_field(m, obj, var, **kwdargs):
    if not ('lat', 'lon') == obj[var].dimensions:
        raise ValueError("expected var to have only lat and lon as dims")
    lats = obj['lat'].data
    lons = obj['lon'].data

    x, y = m(*pylab.meshgrid(lons,lats))
    p = m.imshow(obj[var].data,
                 interpolation = 'gaussian',
                 extent = (np.min(x), np.max(x), np.min(y), np.max(y)),
                 **kwdargs)
    return p

def plot_barbs(m, obj, uvar, vvar, cmap=None):
    if not ('lat', 'lon') == obj[uvar].dimensions:
        raise ValueError("expected var to have only lat and lon as dims")
    if not obj[uvar].dimensions == obj[vvar].dimensions:
        raise ValueError("expected both uvar and vvar to be of same dim")
    lats = obj['lat'].data
    lons = obj['lon'].data

    x, y = m(*pylab.meshgrid(lons,lats))
    U = obj[uvar].data
    V = obj[vvar].data
    return m.barbs(x, y, U, V, flip_barb=False, length=7, cmap=cmap, pivot='middle')

def plot_quiver(m, obj, uvar, vvar, **kwdargs):
    if not ('lat', 'lon') == obj[uvar].dimensions:
        raise ValueError("expected var to have only lat and lon as dims")
    if not obj[uvar].dimensions == obj[vvar].dimensions:
        raise ValueError("expected both uvar and vvar to be of same dim")
    lats = obj['lat'].data
    lons = obj['lon'].data

    x, y = m(*pylab.meshgrid(lons,lats))
    U = obj[uvar].data
    V = obj[vvar].data
    Q = m.quiver(x, y, U, V,
                pivot='mid', color='r', units='dots' )
    #qk = quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$', fontproperties={'weight': 'bold'})
    m.plot(x, y, 'k.')
    return Q

def plot_circle(opts, args):
    """
    Plots what appears as a circle on an orthographic projection on the
    mercator projection
    """
    fig = plt.subplot(1, 2, 1)
    ortho = Basemap(projection='ortho',lon_0=opts.start.lon,lat_0=opts.start.lat,resolution='l')
    center = ortho(opts.start.lon, opts.start.lat)
    plotlib.make_pretty(ortho)
    radius = 3000000
    resol = np.pi/200
    circle_points = [(center[0] + radius * np.cos(x), center[1] + radius * np.sin(x)) for x in np.arange(0, 2*np.pi, step = resol)]
    #lat_lons = map(lambda x: ortho(*x, inverse=True), circle_points)
    circlex = [x[0] for x in circle_points]
    circley = [x[1] for x in circle_points]
    ortho.scatter(circlex, circley, 10, edgecolors='none',zorder=10)

    fig = plt.subplot(1, 2, 2)

    m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20, resolution='l')
    #m = Basemap(projection='sinu',lon_0=0,lat_0=0, resolution='l')
    trans_points = [m(*x) for x in lat_lons]
    xs = [x[0] for x in trans_points]
    ys = [x[1] for x in trans_points]
    m.scatter(xs, ys, 10, edgecolors='none',zorder=10)
    plotlib.make_pretty(m)

    plt.show()
