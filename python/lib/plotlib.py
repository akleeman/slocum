import matplotlib.pyplot as plt

def plot_passage(passage):
    passage = list(passage)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, y)

    plot_route(passage)

    plt.show()

def plot_route(passage):
    passage = list(passage)
    start = passage[0].loc
    end = passage[-1].loc
    print "took: ", passage[-1].time - passage[0].time
    mid = objects.LatLon(0.5*(start.lat + end.lat), 0.5*(start.lon + end.lon))
    m = Basemap(projection='ortho',lon_0=mid.lon,lat_0=mid.lat,resolution='l')
    lons = [x.loc.lon for x in passage]
    lats = [x.loc.lat for x in passage]
    x,y = m(lons,lats)
    # draw colored markers.
    # use zorder=10 to make sure markers are drawn last.
    # (otherwise they are covered up when continents are filled)
    m.scatter(x,y,10,edgecolors='none',zorder=10)
    # map with continents drawn and filled.
    m.drawcoastlines()
    m.fillcontinents(color='coral',lake_color='aqua')
    m.drawcountries()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    m.drawmapboundary(fill_color='aqua')