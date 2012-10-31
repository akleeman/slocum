import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sl.lib import plotlib

def animate_route(passage, proj='lcc'):
    fig = plt.figure(figsize=(16, 8))
    m = plotlib.plot_route(passage)
    plt.ion()
    plt.draw()
    plt.show()
    wx = passage[0].wx
    xx, yy = m(*np.meshgrid(wx['lon'].data, wx['lat'].data))
    is_land = np.vectorize(m.is_land)

    for i, leg in enumerate(passage):
        fig.clf()
        m = plotlib.plot_route(passage)
        plotlib.make_pretty(m)

        x, y = m(leg.course.loc.lon, leg.course.loc.lat)
        m.scatter(x, y, 10, c='r', marker='o', zorder=11)

        wx = leg.wx
        wind_speed = np.sqrt(np.power(wx['uwnd'].data, 2.) + np.power(wx['vwnd'].data, 2.))

        levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
        wind_speed = np.ma.masked_array(wind_speed, is_land(xx, yy))
        waves = m.contour(xx, yy, wx['combined_swell_height'].data,
                          colors='k',
                          levels = [0, 1, 2, 3, 4, 5, 7, 10, 13, 15],
                          zorder=8)
        filled = m.contourf(xx, yy, wind_speed,
                            zorder=7,
                            alpha=1,
                            levels=levels,
                            cmap=plt.cm.jet,
                            extend='both')
        plt.colorbar(filled, drawedges=True)

        labels = plt.clabel(waves, colors='k')
        [lab.set_zorder(10) for lab in labels]
        plt.draw()
        import pdb; pdb.set_trace()
