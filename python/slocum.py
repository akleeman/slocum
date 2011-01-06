#!/usr/bin/python2.6
"""
Some notes:

This cdo command creates a new ncdf file with the maximum value for a day
cdo timselmax,4 analysis_20091201_v11l30flk.nc analysis_20091201_v11l30flk_max.nc

This cdo command creates a new ncdf file with the mean value for a day
cdo timselmean,4 analysis_20091201_v11l30flk.nc analysis_20091201_v11l30flk_max.nc

get ensemble forecasts from:
  ftp://ftp.cdc.noaa.gov/Datasets.other/map/ENS/
or better yet:
  ftp://polar.ncep.noaa.gov/pub/waves/
the file nww3.all contains all sorts of great stuff

hurricane tracks can be downloaded from:
  http://www.nationalatlas.gov/atlasftp.html
see basemap examples for how to plot them

and sea winds are also available at:
  http://nomads.ncdc.noaa.gov/data/seawinds/

netcdf server of gfs data
  http://nomads.ncdc.noaa.gov/thredds/ncss/grid/gens2/201012/20101225/gens-b_2_20101225_0000_384_20.grb2/dataset.html

NCEP moel data: including GFS
  http://motherlode.ucar.edu:9080/thredds/idd/models.html

NetCDF access to subsets of data
  http://www.unidata.ucar.edu/projects/THREDDS/tech/interfaceSpec/GridDataSubsetService.html

High availability ncep data:
  http://nomads.ncep.noaa.gov/

OSCAR ocean currents data
http://www.oscar.noaa.gov/datadisplay/datadownload.htm

REQUIRES:
numpy
scipy
coards
basemap
matplotlib
scientific
pydap

TODO:
- Take into account leeway/waves to create a heading
- Make sure rhumbline calculations are accurate
- Download forecasts on the fly using the fast downloading scheme here: http://www.cpc.ncep.noaa.gov/products/wesley/fast_downloading_grib.html
"""

import sys
import pytz
import numpy as np
import logging
import datetime

from optparse import OptionParser
from matplotlib import pyplot as plt

from lib import objects, navigation, plotlib, fetch, poseidon, iterlib

MAX_BOAT_SPEED = 6
MAX_POINTING = 0.3490658503988659 # 30/180*pi
MIN_WIND_SPEED = 3
REQ_WIND_SPEED = 15
MAX_WIND_SPEED = 45
SECONDS_IN_HOUR = 3600

logging.basicConfig(level=logging.INFO)

def directional_max_speed(deg_off_wind):
    if deg_off_wind > MAX_POINTING:
        return MAX_BOAT_SPEED
    else:
        return (0.5 + 0.5* deg_off_wind / MAX_POINTING) * MAX_BOAT_SPEED

def boat_speed(wind, bearing):
    deg_off_wind = np.abs(wind.dir - bearing)
    if deg_off_wind > np.pi: deg_off_wind = 2*np.pi - deg_off_wind
    if wind.speed > MIN_WIND_SPEED and wind.speed <= REQ_WIND_SPEED:
        speed = np.sqrt((wind.speed - MIN_WIND_SPEED) / (REQ_WIND_SPEED - MIN_WIND_SPEED))
        return directional_max_speed(deg_off_wind) * speed
    elif wind.speed > MIN_WIND_SPEED and wind.speed <= MAX_WIND_SPEED:
        return directional_max_speed(deg_off_wind)
    else:
        return 0

def hours(timedelta):
    return float(timedelta.days * 24 + timedelta.seconds / SECONDS_IN_HOUR)

def simulate_passage(waypoints, start_date, wx_fields):
    """
    Simulates a passage following waypoints having started on a given date.

    waypoints - a list of at least two waypoints, the first of which is the
        start location.
    start_date - the date the passage is started
    wxfunc - a function with signature f(time, lat, lon) that must return a
        pupynere-like object containing at least uwnd, vwnd
    """
    wx_fields = dict(wx_fields)
    waypoints = list(waypoints)
    here = waypoints.pop(0)

    times = wx_fields.keys()
    #tds = [max(now - x, x - now) for x in times]
    start_ind = 0 #tds.index(min(tds))
    end_date = max(times)

    time_iter = iter(sorted(wx_fields.keys()))
    now = time_iter.next()
    soon = time_iter.next()
    dt = soon - now
    try:
        for destination in waypoints:
            while not here == destination:
                # interpolate the weather in wx_fields at the current lat lon
                wx = iterlib.realize(wx_fields[now], dict)
                wx = iterlib.value_map(lambda x: x(here.lat, here.lon), wx)
                uwnd = wx.pop('uwnd')
                vwnd = wx.pop('vwnd')

                # determine the bearing (following a rhumbline) between here and the end
                bearing = navigation.rhumbline_bearing(here, destination)
                # get the wind and use that to compute the boat speed
                wind = objects.Wind(uwnd, vwnd)
                speed = max(boat_speed(wind, bearing), 1.0)
                course = objects.Course(here, speed, bearing, bearing)
                # given our speed how far can we go in one timestep?
                distance = speed * hours(dt)
                remaining = navigation.rhumbline_distance(here, destination)
                if distance > remaining:
                    here = destination
                    required_time = int(hours(dt) * SECONDS_IN_HOUR * remaining / distance)
                    now = now + datetime.timedelta(seconds=required_time)
                    distance = remaining
                else:
                    # and once we know how far, where does that put us in terms of lat long
                    here = navigation.rhumbline_path(here, bearing)(distance)
                    now = soon
                    soon = time_iter.next()
                dt = soon - now
                logging.info('wind: %4s (%4.1f) @ %6.1f knots \t %6.1f miles in %4.1f hours @ %6.1f knots'
                             % (wind.readable, wind.dir, wind.speed, distance, hours(dt), speed))
                yield objects.Leg(course, now, wind, distance, wx)
    except StopIteration:
        logging.error("Ran out of data!")

def historical_passages(waypoints, start_date, first_year=None, last_year=None):
    first_year = 2000 if not first_year else first_year
    last_year = 2009 if not last_year else last_year

    start_day = start_date.timetuple().tm_mday
    start_mon = start_date.timetuple().tm_mon

    for year in range(first_year, last_year + 1):
        date = datetime.datetime(year, start_mon, start_day)
        yield simulate_passage(waypoints, date)

def optimal_passage(start, end, start_date=None, resol=50):
    c1 = objects.LatLon(start.lat, end.lon)
    c2 = objects.LatLon(end.lat, start.lon)

    if not start_date:
        start_date = datetime.datetime(2005, 1, 1)

    def get_time(x):
        passage = list(simulate_passage([start, x, end], start_date))
        passage_time = hours(passage[-1].time - passage[0].time)
        return passage_time

    waypoints = [objects.LatLon(x*c1.lat + (1.-x)*c2.lat, x*c1.lon + (1.-x)*c2.lon) for x in np.arange(0., 1., step=1./resol)]
    return waypoints[np.argmin([get_time(x) for x in waypoints])]

def summarize_passage(passage):
    ret = {}
    passage = list(passage)
    dt = (passage[-1].time - passage[0].time)
    ret['days'] = dt.days + dt.seconds/(24*SECONDS_IN_HOUR)
    wind = [x.wind.speed for x in passage]
    ret.update({'min_wind':np.min(wind), 'max_wind':np.max(wind), 'avg_wind':np.mean(wind)})
    dist = [x.distance for x in passage]
    ret.update({'min_dist':np.min(dist), 'max_dist':np.max(wind), 'avg_dist':np.mean(wind)})
    return ret

def handle_plot(opts, args):
    passage = simulate_passage([opts.start, opts.end], start_date=opts.start_date)
    plotlib.plot_passage(list(passage))

def handle_optimal(opts, args):
    waypoint = optimal_passage(opts.start, opts.end, opts.start_date)
    passage = simulate_passage([opts.start, waypoint, opts.end], opts.start_date)
    plotlib.plot_passage(list(passage))

def handle_historical(opts, args):
    hist_passages = list(historical_passages([opts.start, opts.end], opts.start_date))
    hist_summary = [summarize_passage(x)for x in hist_passages]

    keys = hist_summary[0].keys()
    n = len(keys)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    for i, k in enumerate(keys):
        fig = plt.subplot(rows, cols, i)
        fig.hist([x[k] for x in hist_summary])
        plt.title(k)
    plt.show()

def main():

    p = OptionParser(usage="""%%prog [options]
    Slocum -- A tool for ocean passage planning

    Joshua Slocum (February 20, 1844 -on or shortly after November 14, 1909)
    was a Canadian-American seaman and adventurer, a noted writer, and the first
    man to sail single-handedly around the world. In 1900 he told the story of
    this in Sailing Alone Around the World. He disappeared in November 1909
    while aboard his boat, the Spray. (wikipedia)
    """)
    p.add_option("", "--fetch-ccmp", default=False, action="store_true",
        help="download and process the cross calibrated multi platform dataset")
    p.add_option("", "--start", default=None, action="store",
        help="the start location ie.  --start=lat,lon")
    p.add_option("", "--end", default=None, action="store",
        help="the end location ie.  --end=lat,lon")
    p.add_option("", "--start-date", default=None, action="store")
    p.add_option("", "--hist", default=False, action="store_true")
    p.add_option("", "--plot", default=False, action="store_true")
    p.add_option("", "--optimal", default=False, action="store_true")
    p.add_option("", "--forecast", default=False, action="store_true")
    p.add_option("-v", "--verbose", default=False, action="store_true")

    opts, args = p.parse_args()

    if opts.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if opts.fetch_ccmp:
        return fetch.fetch_ccmp()

    if opts.start:
        opts.start = objects.LatLon(*[float(x) for x in opts.start.split(',')])
    else:
        opts.start = objects.LatLon(36.625, -121.9) # sf

    if opts.end:
        opts.end = objects.LatLon(*[float(x) for x in opts.end.split(',')])
    else:
        opts.end = objects.LatLon(19.79, -154.76) # hawaii

    if opts.start_date:
        opts.start_date = datetime.datetime.strptime(opts.start_date, '%Y-%m-%d')
    else:
        opts.start_date = datetime.datetime.now()

    eastern = pytz.timezone('US/Pacific')
    opts.start_date = eastern.localize(opts.start_date)

    if opts.hist:
        return handle_historical(opts, args)

    if opts.optimal:
        return handle_optimal(opts, args)

    if opts.plot:
        return handle_plot(opts, args)

    if opts.forecast:

        passages = []
        for fcst in poseidon.forecast_weather(opts.start_date, opts.start, opts.end):
            passages.append(list(simulate_passage([opts.start, opts.end], start_date=opts.start_date, wx_fields=fcst)))
#        for wx in poseidon.historical_weather(opts.start_date):
#            passages.append(list(simulate_passage([opts.start, opts.end], start_date=opts.start_date, wx_fields=wx)))
        plotlib.plot_passages(passages, 'combined_swell_height')
        return 0
    p.error("slocum completed exactly what you told it to do ... nothing.")

if __name__ == "__main__":
    sys.exit(main())
