import numpy as np
import coards

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from matplotlib.pylab import meshgrid

import wx.objects.conventions as conv

from wx.lib import datelib
from wx.objects import objects

def make_pretty(m):
    # map with continents drawn and filled.
    m.drawcoastlines()
    m.fillcontinents(color='green',lake_color='aqua')
    m.drawcountries()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    m.drawmapboundary(fill_color='aqua')

def plot_passages(passages, etc_var=None):
    if conv.ENSEMBLE in passages.variables and passages.dimensions[conv.ENSEMBLE] > 1:
        passages = [y for x, y in passages.iterator(conv.ENSEMBLE)]
    else:
        passages = [passages]
    passage = passages[0]
    if conv.STEP in passage.dimensions:
        passage = passage.view(slice(0, passage[conv.NUM_STEPS].data), conv.STEP)
    times = datelib.from_udvar(passage[conv.TIME])
    units = min(times)[0].strftime('days since %Y-%m-%d %H:00:00')
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(1, 3, 1)
    plot_route(passage)

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_title("Boat Speed")
    ax2.set_ylim([0, 7])
    ax3 = fig.add_subplot(3, 3, 3, sharex=ax2)
    ax3.set_title("Wind Speed")
    ax4 = fig.add_subplot(3, 3, 5, sharex=ax2)
    ax4.set_title("Relative Wind Dir")
    ax4.set_ylim([0, np.pi])
    ax5 = fig.add_subplot(3, 3, 6, sharex=ax2)
    ax5.set_title("Motor On")
    ax5.set_ylim([0, 1])
    if etc_var:
        ax5 = fig.add_subplot(3, 3, 6, sharex=ax2)
        ax5.set_title(etc_var)
    for passage in passages:
        if conv.STEP in passage.dimensions:
            passage = passage.view(slice(0, passage[conv.NUM_STEPS].data), conv.STEP)
        times = datelib.from_udvar(passage[conv.TIME])
        time = np.array([coards.to_udunits(x[0], units) for x in times])
        if max(time) <= 2:
            # if there are less than 2 days in this passage use hours
            time = time * 24
            units = units.replace('days', 'hours')
        ax2.plot(time, passage[conv.SPEED].data)
        wind = [objects.Wind(u, v) for u, v in
                zip(passage[conv.UWND].data, passage[conv.VWND].data)]
        wind_speed = [x.speed for x in wind]
        ax3.plot(time, wind_speed)
        ax3.set_ylim([0, np.max(wind_speed) + 5])
        def rel(x):
            if x < np.pi:
                return x
            else:
                return 2*np.pi - x
        rel_wind = [rel(np.abs(heading - w.dir)) for heading, w in zip(passage[conv.HEADING], wind)]
        ax4.plot(time, rel_wind)
        ax5.plot(time, passage[conv.MOTOR_ON].data)
        ax5.set_ylim([-0.1, 1.1])
    fig.suptitle('X axis shows %s' % units)
    plt.show()

def add_wind_contours(m, wx):
        wind_speed = np.sqrt(np.power(wx['uwnd'].data, 2.) + np.power(wx['vwnd'].data, 2.))
        levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
        lines = m.contour(xx, yy, speed,
                          colors='k',
                          zorder=7,
                          levels=levels)
        filled = m.contourf(xx, yy, speed,
                            zorder=7,
                            alpha=0.5,
                            levels=lines.levels,
                            cmap=plt.cm.jet,
                            extend='both')
        plt.colorbar(filled, drawedges=True)
        plt.draw()

def plot_passage(passage):
    plot_passages([passage])

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

def plot_map(passage, proj='lcc', pad=0.25):
    start_lat = np.mean(passage[conv.LAT].data[0])
    start_lon = np.mean(passage[conv.LON].data[0])
    if conv.STEP in passage.dimensions:
        end_lat = np.mean([x[conv.LAT].data[x[conv.NUM_STEPS].data - 1]
                           for _, x in passage.iterator(conv.ENSEMBLE)])
        end_lon = np.mean([x[conv.LON].data[x[conv.NUM_STEPS].data - 1]
                           for _, x in passage.iterator(conv.ENSEMBLE)])
    else:
        end_lat = np.mean(passage[conv.LAT].data[-1])
        end_lon = np.mean(passage[conv.LON].data[-1])
    mid = objects.LatLon(0.5*(start_lat + end_lat), 0.5*(start_lon + end_lon))

    lat_deg = max(np.abs(start_lat - end_lat), 2)
    lon_deg = max(np.abs(start_lon - end_lon), 2)

    llcrnrlon=min(start_lon, end_lon)-pad*lon_deg
    llcrnrlat=min(start_lat, end_lat)-pad*lat_deg
    urcrnrlon=max(start_lon, end_lon)+pad*lon_deg
    urcrnrlat=max(start_lat, end_lat)+pad*lat_deg

    m = Basemap(projection=proj,
                lon_0=mid.lon,
                lat_0=mid.lat,
                llcrnrlon=llcrnrlon,
                llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon,
                urcrnrlat=urcrnrlat,
                rsphere=(6378137.00,6356752.3142),
                area_thresh=1000.,
                width=np.abs(start_lon - end_lon),
                height=np.abs(start_lat - end_lat),
                resolution='l')

    make_pretty(m)
    return m

def plot_route(passage, proj='lcc'):
    m = plot_map(passage, proj=proj)
    lats = passage[conv.LAT].data
    lons = passage[conv.LON].data
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

    m = plot_map(first_route, proj=proj)

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

def plot_field(field, proj='lcc'):

    field = field['uwnd']
    lats, lons = field.dims

    start = objects.LatLon(min(lats), min(lons))
    end = objects.LatLon(max(lats), max(lons))
    mid = objects.LatLon(0.5*(start.lat + end.lat), 0.5*(start.lon + end.lon))

    llcrnrlon=min(start.lon, end.lon)-5
    llcrnrlat=min(start.lat, end.lat)-5
    urcrnrlon=max(start.lon, end.lon)+5
    urcrnrlat=max(start.lat, end.lat)+5

    m = Basemap(projection=proj,
                lon_0=mid.lon,
                lat_0=mid.lat,
                llcrnrlon=llcrnrlon,
                llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon,
                urcrnrlat=urcrnrlat,
                rsphere=(6378137.00, 6356752.3142),
                area_thresh=1000.,
                width=np.abs(start.lon - end.lon),
                height=np.abs(start.lat - end.lat),
                resolution='l')

    x, y = m(*meshgrid(lons,lats))
    m.contour(x, y, field.data)
    # map with continents drawn and filled.
    make_pretty(m)
    plt.show()

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
