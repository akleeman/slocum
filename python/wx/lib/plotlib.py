import numpy as np
import coards
import logging

import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)

try:
    from mpl_toolkits.basemap import Basemap
except:
    logging.debug("not loading basemap")

from bisect import bisect
from matplotlib import pylab, cm

import wx.objects.conventions as conv

from wx.lib import datelib
from wx.objects import objects

def plot_wind(fcst, **kwdargs):
    hist_fig = plt.figure(figsize=(8, 8))
    fig = plt.figure(figsize=(16, 10))
    map_axis= fig.add_subplot(1,1,1)
    hist_axis = fig.add_subplot(1,1,1)
    lons = fcst['lon'].data
    lats = fcst['lat'].data
    ll = objects.LatLon(min(lats), min(lons))
    ur = objects.LatLon(max(lats), max(lons))

    h = hist_axis.hist(fcst['wind_speed'].data.flatten())

    for i, (_, fc) in enumerate(fcst.iterator(conv.TIME)):
        m = plot_map(ll, ur)
        m.fillcontinents(color='green',lake_color='white')
        m.drawparallels(lats, zorder=5)
        m.drawmeridians(lons, zorder=5)
        x, y = m(*pylab.meshgrid(lons,lats))
        fc_time = datelib.from_udvar(fc[conv.TIME])[0]
        fc.squeeze(conv.TIME)
        ens_axis = fc['wind_dir'].dimensions.index(conv.ENSEMBLE)
        ax = wind(x, y, fc['wind_dir'].data, fc['wind_speed'].data)
        ax.set_title(fc_time.strftime('%A %B %d at %H:%M GMT'))
        ax.set_xticks(x[-1, :])
        ax.set_xticklabels(['%.0fE' % z for z in lons])
        ax.set_yticks(y[:, 0])
        ax.set_yticklabels(['%.0fN' % z for z in lats])

        def onpress(event):
            lon, lat = m(event.xdata, event.ydata, inverse=True)
            print lat, lon
            lat_ind = np.argmin(np.abs(fc['lat'].data - lat))
            grid = fc.take([lat_ind], 'lat')
            grid.squeeze('lat')
            lon_ind = np.argmin(np.abs(fc['lon'].data - lon))
            grid = grid.take([lon_ind], 'lon')
            grid.squeeze('lon')
            (grid['wind_speed'].data.flatten())


        fig.canvas.mpl_connect('button_press_event', onpress)
        plt.show()
        import pdb; pdb.set_trace()
#        plt.pause(2)
#        plt.clf()
    return ax

def wind(x, y, wind_dir, wind_speed, ax=None, scale=1.):
    if not (wind_dir.shape == wind_speed.shape):
        raise ValueError("expected all arrays to be the same size")

    radius = scale * 0.45 * np.min(np.diff(x))
    x = x.flatten()
    y = y.flatten()

    if not ax:
        ax = plt.subplot(1,1,1)
    n = 5.
    for i in range(1):
        circles = [plt.Circle((ix, iy), radius=radius * np.power((n - i) / n, 0.8),
                              edgecolor='k', facecolor='w', zorder=10, alpha=0.6) for ix, iy in zip(x, y)]
        [ax.add_patch(c) for c in circles]
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    max_wind = np.max(wind_speed)
    for (wd, ws) in zip(wind_dir, wind_speed):
        wd = wd.flatten()
        ws = ws.flatten()
        ws_scale = radius # * np.sqrt(ws / max_wind)
        min_x = x + ws_scale * np.sin(wd - np.pi/16.)
        min_y = y + ws_scale * np.cos(wd - np.pi/16.)
        max_x = x + ws_scale * np.sin(wd + np.pi/16.)
        max_y = y + ws_scale * np.cos(wd + np.pi/16.)
        xn = x.shape[0]
        xs = np.concatenate([x, min_x, max_x])
        ys = np.concatenate([y, min_y, max_y])
        triangles = [(i, i + xn, i + 2*xn) for i in range(xn)]
        ws = ws.repeat(3).reshape(xn, 3).T.flatten()
        ax.tripcolor(xs, ys, triangles, ws, shading='flat',
                     edgecolors='none', facecolors='r', alpha=0.7, zorder=100)
    return ax

def tmp_wind(x, y, min_dir, max_dir, wind_speed, scale=1.):
    if not (x.shape == max_dir.shape and
            y.shape == max_dir.shape and
            min_dir.shape == max_dir.shape):
        raise ValueError("expected all arrays to be the same size")

    import pdb; pdb.set_trace()
    min_dir = min_dir.flatten()
    max_dir = max_dir.flatten()
    mid_dir = 0.5 * (min_dir + max_dir)
    min_x = x + radius * np.sin(min_dir - np.pi/32.)
    min_y = y + radius * np.cos(min_dir - np.pi/32.)
    mid_x = x + radius * np.sin(mid_dir - np.pi/32.)
    mid_y = y + radius * np.sin(mid_dir - np.pi/32.)
    max_x = x + radius * np.sin(max_dir + np.pi/32.)
    max_y = y + radius * np.cos(max_dir + np.pi/32.)

    n = x.shape[0]
    xs = np.concatenate([x, min_x, max_x])
    ys = np.concatenate([y, min_y, max_y])
    triangles = [(i, i + n, i + 2*n) for i in range(n)]
    wind_speed = wind_speed.repeat(3).reshape(n, 3).T.flatten()
    plt.tripcolor(xs, ys, triangles, wind_speed, shading='faceted')
    plt.plot(x, y, 'k.')
    plt.show()
#x = xy[:,0]*180/3.14159
#y = xy[:,1]*180/3.14159
#x0 = -5
#y0 = 52
#z = np.exp(-0.01*( (x-x0)*(x-x0) + (y-y0)*(y-y0) ))
#
#triangles = np.asarray([
#    [67,66, 1],[65, 2,66],[ 1,66, 2],[64, 2,65],[63, 3,64],[60,59,57],
#    [ 2,64, 3],[ 3,63, 4],[ 0,67, 1],[62, 4,63],[57,59,56],[59,58,56],
#    [61,60,69],[57,69,60],[ 4,62,68],[ 6, 5, 9],[61,68,62],[69,68,61],
#    [ 9, 5,70],[ 6, 8, 7],[ 4,70, 5],[ 8, 6, 9],[56,69,57],[69,56,52],
#    [70,10, 9],[54,53,55],[56,55,53],[68,70, 4],[52,56,53],[11,10,12],
#    [69,71,68],[68,13,70],[10,70,13],[51,50,52],[13,68,71],[52,71,69],
#    [12,10,13],[71,52,50],[71,14,13],[50,49,71],[49,48,71],[14,16,15],
#    [14,71,48],[17,19,18],[17,20,19],[48,16,14],[48,47,16],[47,46,16],
#    [16,46,45],[23,22,24],[21,24,22],[17,16,45],[20,17,45],[21,25,24],
#    [27,26,28],[20,72,21],[25,21,72],[45,72,20],[25,28,26],[44,73,45],
#    [72,45,73],[28,25,29],[29,25,31],[43,73,44],[73,43,40],[72,73,39],
#    [72,31,25],[42,40,43],[31,30,29],[39,73,40],[42,41,40],[72,33,31],
#    [32,31,33],[39,38,72],[33,72,38],[33,38,34],[37,35,38],[34,38,35],
#    [35,37,36] ])
#
## Rather than create a Triangulation object, can simply pass x, y and triangles
## arrays to tripcolor directly.  It would be better to use a Triangulation object
## if the same triangulation was to be used more than once to save duplicated
## calculations.
#plt.figure()
#plt.gca().set_aspect('equal')
#plt.tripcolor(x, y, triangles, z, shading='faceted')
#plt.colorbar()
#plt.title('tripcolor of user-specified triangulation')
#plt.xlabel('Longitude (degrees)')
#plt.ylabel('Latitude (degrees)')

def plot_variance(fcst):
    plt.figure(figsize=(16, 10))
    lons = fcst['lon'].data
    lats = fcst['lat'].data
    ll = objects.LatLon(min(lats), min(lons))
    ur = objects.LatLon(max(lats), max(lons))
    m = plot_map(ll, ur)
    m.fillcontinents(color='green',lake_color='white')

    for i, (_, fc) in enumerate(fcst.iterator(conv.TIME)):
        var = fc.take([0], conv.ENSEMBLE)
        var['wind_speed'].data[:] = np.var(fc['wind_speed'].data, axis=0)
        var = var.squeeze(conv.TIME)
        var = var.squeeze(conv.ENSEMBLE)

        mean = fc.take([0], conv.ENSEMBLE)
        mean['wind_speed'].data[:] = np.mean(fc['wind_speed'].data, axis=0)
        mean = mean.squeeze(conv.TIME)
        mean = mean.squeeze(conv.ENSEMBLE)
        if i == 0:
            field = plot_field(m, var, 'wind_speed', cmap=cm.Blues, alpha=0.8)
            plt.clim()
            barbs = plot_barbs(m, mean, 'uwnd', 'vwnd', cmap=cm.hot_r)
        else:
            field.set_data(var['wind_speed'].data)
            U = mean['uwnd'].data
            V = mean['vwnd'].data
            barbs[1].set_UVC(U, V, np.sqrt(U*U + V*V))
        plt.pause(0.5)

def plot_forecast(fcst):

    iter_var = conv.ENSEMBLE
    slice_var = conv.TIME
    fcst = fcst.take([0], slice_var)
    fcst = fcst.squeeze(slice_var)
    plt.figure(figsize=(16, 10))
    for i, (_, fc) in enumerate(fcst.iterator(iter_var)):
        fc = fc.squeeze(iter_var)
        if i == 0:
            lons = fcst['lon'].data
            lats = fcst['lat'].data
            ll = objects.LatLon(min(lats), min(lons))
            ur = objects.LatLon(max(lats), max(lons))
            m = plot_map(ll, ur)
            #m.drawparallels(lats)
            #m.drawmeridians(lons)
            p = plot_quiver(m, fc, 'uwnd', 'vwnd')[1]
            #p = plot_field(m, fc, 'wind_speed')
            #plt.clim()
        else:
            #p.set_data(fc['wind_speed'].data)
            U = fc['uwnd'].data
            V = fc['vwnd'].data
            p.set_UVC(U, V, np.sqrt(U*U + V*V))
        plt.pause(0.5)
    #plot_barbs(m, fcst, 'uwnd', 'vwnd')
    #plt.imshow(speed,
    #           extent = (min(lons), max(lons), min(lats), max(lats)))
#    bins = np.array([1., 3., 6., 10., 16., 21., 27., 33., 40., 47., 55., 63., np.inf])
#    colors = ['#a1eeff', # light blue
#              '#42b1e5', # darker blue
#              '#60fd4b', # green
#              '#1cea00', # yellow-green
#              '#fbef36', # yellow
#              '#fbc136', # orange
#              '#ff4f02', # red
#              '#ff0e02', # darker-red
#              '#ff00c0', # red-purple
#              '#d925ac', # purple
#              '#b30d8a', # dark purple
#              '#000000', # black
#              ]

def make_pretty(m, ocean_color='#dcdcdc'):
    # map with continents drawn and filled.
    m.drawcoastlines()
    m.drawlsmask(land_color='green',ocean_color=ocean_color,lakes=True)
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

def plot_passage_map(passage, proj='lcc', pad=0.25):
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

def plot_map(ll, ur, proj='lcc', pad=0.25):
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
