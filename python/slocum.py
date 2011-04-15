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

from wx import poseidon, spray
from wx.lib import plotlib, emaillib, griblib
from wx.objects import objects

def getdata(opts, args):
    if opts.grib:
        return list(griblib.degrib(opts.grib))
    if opts.download:
        return list(poseidon.forecast_weather(opts.start_date, opts.start, opts.end))

def handle_simulate(opts, args):
    data = getdata(opts, args)
    waypoints = [opts.start, opts.end]
    passages = list(spray.passage(waypoints, opts.start_date, data))
    return 1

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
    p.add_option("", "--grib", default=None, action="store",
                 help="A grib file containing forecasts")
    p.add_option("", "--download", default=None, action="store_true",
                 help="Download forecasts from UCAR")

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

    return handle_simulate(opts, args)

    p.error("slocum completed exactly what you told it to do ... nothing.")

if __name__ == "__main__":
    sys.exit(main())
