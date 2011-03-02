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
import coards
import logging
import datetime

from optparse import OptionParser
from matplotlib import pyplot as plt

from wx import poseidon
from wx.lib import objects, navigation, plotlib, fetch, iterlib, emaillib

MAX_BOAT_SPEED = 6
MAX_POINTING = 0.3490658503988659 # 30/180*pi
MIN_WIND_SPEED = 3
REQ_WIND_SPEED = 15
MAX_WIND_SPEED = 35
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

def simulate_passage(waypoints, start_date, forecasts):
    """
    Simulates a passage following waypoints having started on a given date.

    waypoints - a list of at least two waypoints, the first of which is the
        start location.
    start_date - the date the passage is started
    wxfunc - a function with signature f(time, lat, lon) that must return a
        pupynere-like object containing at least uwnd, vwnd
    """
    waypoints = list(waypoints)
    here = waypoints.pop(0)

    fcst = forecasts[1]
    fcsts = fcst.iterator(dim='time')
    def next_time():
        t, y = fcsts.next()
        assert t.data.size == 1
        return coards.from_udunits(t.data[0], t.units), y

    now, wx = next_time()
    soon, next_wx = next_time()
    dt = soon - now
    try:
        for destination in waypoints:
            while not here == destination:
                # interpolate the weather in wx_fields at the current lat lon
                local_wx = wx.interpolate(lat=here.lat, lon=here.lon)
                uwnd = np.asscalar(local_wx['uwnd'].data)
                vwnd = np.asscalar(local_wx['vwnd'].data)
                # determine the bearing (following a rhumbline) between here and the end
                bearing = navigation.rhumbline_bearing(here, destination)
                # get the wind and use that to compute the boat speed
                wind = objects.Wind(uwnd, vwnd)
                speed = max(boat_speed(wind, bearing), 1.0)
                course = objects.Course(here, speed, bearing, bearing)
                rel_wind = np.abs(course.heading - wind.dir)
                if rel_wind > np.pi:
                    rel_wind = 2.*np.pi - rel_wind
                # given our speed how far can we go in one timestep?
                distance = speed * hours(dt)
                remaining = navigation.rhumbline_distance(here, destination)
                if distance > remaining:
                    here = destination
                    required_time = int(hours(dt) * SECONDS_IN_HOUR * remaining / distance)
                    now = now + datetime.timedelta(seconds=required_time)
                    dt = soon - now
                    distance = remaining
                else:
                    # and once we know how far, where does that put us in terms of lat long
                    here = navigation.rhumbline_path(here, bearing)(distance)
                    now, wx = soon, next_wx
                    soon, next_wx = next_time()
                dt = soon - now
                logging.debug('wind: %4s (%4.1f) @ %6.1f knots \t %6.1f miles in %4.1f hours @ %6.1f knots'
                             % (wind.readable, wind.dir, wind.speed, distance, hours(dt), speed))

                yield objects.Leg(course, now, wind, distance, rel_wind, wx)
    except StopIteration:
        logging.error("Ran out of data!")

def simulate_passages(waypoints, start_date, forecasts):
    def passage(fcsts):
        return list(simulate_passage(waypoints, start_date=start_date, forecasts=fcsts))
    return [passage(field) for field in forecasts]

def historical_passages(waypoints, start_date, first_year=None, last_year=None):
    if not first_year:
        first_year = 2000
    if not last_year:
        last_year = 2009

    start_day = start_date.timetuple().tm_mday
    start_mon = start_date.timetuple().tm_mon

    for year in range(first_year, last_year + 1):
        date = datetime.datetime(year, start_mon, start_day)
        yield simulate_passage(waypoints, date)

def optimal_passage(start, end, start_date, forecasts, resol=5):
    # get the corners
    c1 = objects.LatLon(start.lat, end.lon)
    c2 = objects.LatLon(end.lat, start.lon)
    forecasts = list(forecasts)

    def issafe(route):
        "returns a boolean indicating the route was a safe one"
        return np.max([x['max_wind'] for x in route]) <= MAX_WIND_SPEED

    def route(x):
        "Returns the passage summaries for a route through x"
        passages = simulate_passages([start, x, end], start_date, forecasts)
        summaries = [summarize_passage(passage) for passage in passages]
        if issafe(summaries):
            return summaries
        else:
            return None
        return summaries

    waypoints = [objects.LatLon(x*c1.lat + (1.-x)*c2.lat, x*c1.lon + (1.-x)*c2.lon) for x in np.arange(0., 1., step=1./resol)]
    routes = [(x, route(x)) for x in waypoints]
    avg_times = np.mean([np.mean([x['hours'] for x in route]) for x, route in routes])
    avg_distances = np.mean([np.mean([x['distance'] for x in route]) for x, route in routes])
    def idealness(route):
        "returns a scalar factor representing the idealness of a route, smaller is better"
        time = np.mean([x['hours'] for x in route])
        dist = np.mean([x['distance'] for x in route])
        pct_upwind = np.mean([x['pct_upwind'] for x in route])
        return (time - avg_times)/avg_times + (avg_distances/dist - 1.) + pct_upwind
    idealness = [idealness(route) for x, route in routes]

    return routes[np.argmin(idealness)][0]

def summarize_passage(passage):
    ret = {}
    passage = list(passage)

    ret['hours'] = hours(passage[-1].time - passage[0].time)
    ret['distance'] = navigation.rhumbline_distance(passage[-1].course.loc,
                                                    passage[0].course.loc)
    wind = [x.wind.speed for x in passage]
    ret.update({'min_wind':np.min(wind), 'max_wind':np.max(wind), 'avg_wind':np.mean(wind)})
    dist = [x.distance for x in passage]
    ret.update({'min_dist':np.min(dist), 'max_dist':np.max(wind), 'avg_dist':np.mean(wind)})
    ret['pct_upwind'] = [x.rel_wind_dir < np.pi/4 for x in passage]
    return ret

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

def handle_forecasts(opts, args):
    """
    handles the simulation of passages based off forecasts
    """
    forecasts = list(poseidon.forecast_weather(opts.start_date, opts.start, opts.end))
    if opts.optimal:
        mid = optimal_passage(opts.start, opts.end, opts.start_date, forecasts)
        waypoints = [opts.start, mid, opts.end]
    else:
        waypoints = [opts.start, opts.end]

    forecasts = [forecasts[0]]
    print "HACK"
    passages = simulate_passages(waypoints, opts.start_date, forecasts)

    for passage in passages:
        from lib import animatelib
        animatelib.animate_route(passage)
        import pdb; pdb.set_trace()
        plotlib.plot_route(passage)
    #plotlib.plot_passages(passages)#, 'combined_swell_height')
    return 0

def main(opts=None, args=None):
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
    p.add_option("", "--optimal", default=False, action="store_true")
    p.add_option("", "--forecast", default=False, action="store_true")
    p.add_option("-v", "--verbose", default=False, action="store_true")
    p.add_option("", "--small", default=False, action="store_true")

    # if the first argument is "email" the arguments will actually be
    # pulled from an MIMEText email piped through stdin
    if len(sys.argv) > 1 and sys.argv[1] == "email":
        email_args = emaillib.args_from_email(sys.stdin.read())
        opts, args = p.parse_args(args=email_args)
    else:
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

    if opts.forecast:
        return handle_forecasts(opts, args)

    p.error("slocum completed exactly what you told it to do ... nothing.")

if __name__ == "__main__":
    sys.exit(main())
