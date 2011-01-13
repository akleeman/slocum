import numpy as np
import coards
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from matplotlib.pylab import meshgrid

from lib import objects

def make_pretty(m):
    # map with continents drawn and filled.
    m.drawcoastlines()
    m.fillcontinents(color='coral',lake_color='aqua')
    m.drawcountries()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    m.drawmapboundary(fill_color='aqua')

def plot_passages(passages, etc_var=None):
    passage = passages[0]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(1, 3, 1)
    plot_route(passage)

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_title("Distance")
    ax3 = fig.add_subplot(3, 3, 3, sharex=ax2)
    ax3.set_title("Wind Speed")
    ax4 = fig.add_subplot(3, 3, 5, sharex=ax2)
    ax4.set_title("Relative Wind Dir")
    ax5 = fig.add_subplot(3, 3, 6, sharex=ax2)
    if etc_var:
        ax5 = fig.add_subplot(3, 3, 6, sharex=ax2)
        ax5.set_title(etc_var)
    for passage in passages:
        times = [x.time for x in passage]
        units = min(times).strftime('days since %Y-%m-%d')
        days = [coards.to_udunits(x, units) for x in times]
        ax2.plot(np.array(days), np.array([x.distance for x in passage]))
        ax3.plot(days, [x.wind.speed for x in passage])
        def rel(x):
            if x < np.pi:
                return x
            else:
                return 2*np.pi - x
        rel_wind = [rel(np.abs(x.course.heading - x.wind.dir)) for x in passage]
        ax4.plot(days, rel_wind)
#        if etc_var:
            #ax5.plot(days, [x.etc[etc_var] if etc_var in x.etc else np.nan for x in passage])

    plt.show()

def plot_passage(passage):
    passage = list(passage)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(1, 2, 1)
    plot_route(passage)

    times = [x.time for x in passage]
    units = min(times).strftime('days since %Y-%m-%d')
    days = [coards.to_udunits(x, units) for x in times]

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(np.array(days), np.array([x.distance for x in passage]))
    plt.title("Distance")

    ax3 = fig.add_subplot(3, 2, 4, sharex=ax2)
    ax3.plot(days, [x.wind.speed for x in passage])
    plt.title("Wind Speed")

#    ax4 = fig.add_subplot(3, 2, 6, sharex=ax2)
#    ax4.plot(days, [x.storm_wind for x in passage])
#    plt.title("Storm winds")
    plt.show()

def plot_route(passage, proj='lcc'):
    passage = list(passage)
    start = passage[0].course.loc
    end = passage[-1].course.loc
    print "took: ", passage[-1].time - passage[0].time
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
                rsphere=(6378137.00,6356752.3142),
                area_thresh=1000.,
                width=np.abs(start.lon - end.lon),
                height=np.abs(start.lat - end.lat),
                resolution='l')

    lons = [x.course.loc.lon for x in passage]
    lats = [x.course.loc.lat for x in passage]
    x,y = m(lons,lats)
    # draw colored markers.
    # use zorder=10 to make sure markers are drawn last.
    # (otherwise they are covered up when continents are filled)
    m.scatter(x,y,10,edgecolors='none',zorder=10)
    # map with continents drawn and filled.
    make_pretty(m)

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
    lat_lons = map(lambda x: ortho(*x, inverse=True), circle_points)
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
