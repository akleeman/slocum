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

import os
import sys
import pytz
import zlib
import numpy as np
import base64
import logging
import datetime
import tempfile

from optparse import OptionParser
from matplotlib import pyplot as plt

import sl.objects.conventions as conv

from sl import poseidon, spray
from sl.lib import plotlib, emaillib, griblib, datelib, tinylib
from sl.objects import objects, core

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))

def weather(opts, args):
    if opts.grib:
        return griblib.degrib(opts.grib)
    if opts.fcst:
        f = open(opts.fcst, 'r')
        fcst = tinylib.from_beaufort(zlib.decompress(f.read()))
        return [[y] for x, y in fcst.iterator(conv.ENSEMBLE)]
    if opts.download or opts.recent:
        ll_lat = min([x.lat for x in opts.waypoints])
        ll_lon = min([x.lon for x in opts.waypoints])
        ur_lat = max([x.lat for x in opts.waypoints])
        ur_lon = max([x.lon for x in opts.waypoints])
        return poseidon.forecast_weather(opts.start_date,
                                         objects.LatLon(ur_lat, ur_lon),
                                         objects.LatLon(ll_lat, ll_lon),
                                         recent=opts.recent)
    else:
        raise ValueError("no data source specified (--grib or --download")

def handle_simulate(opts, args):
    wx = list(weather(opts, args))
    waypoints = opts.waypoints
    if len(wx) > 1:
        passages = [spray.passage(waypoints, opts.start_date, ens, fast=opts.fast) for ens in wx]
        plotlib.plot_passages(objects.normalize_ensemble_passages(passages))
    else:
        passage = list(spray.passage(waypoints, opts.start_date, wx, fast=opts.fast))
        plotlib.plot_passage(passage)
    return 0

def handle_when(opts, args):
    wx = list(weather(opts, args))
    if opts.ensemble:
        wx = [gfs for gfs, nww3 in wx]
    else:
        wx = [weather(opts, args).next()[0]]
    waypoints = opts.waypoints
    def simulate(start_date):
        return spray.ensemble_passage(waypoints, start_date, weather=wx, fast=False)

    start_dates = [opts.start_date + datetime.timedelta(seconds=i*12*3600) for i in xrange(8)]
    iterroutes = [(waypoints, simulate(d)) for d in start_dates]
    optimal_passages = spray.optimal_passage(iterroutes)
    print "best departure", min(datelib.from_udvar(optimal_passages[conv.TIME]))

    passages = spray.normalize_ensemble_passages([x[1] for x in iterroutes])
    summaries = ((sd, spray.summarize_passage(p)) for sd, (_, p) in
                 zip(start_dates, passages.iterator(conv.ENSEMBLE)))
    sd, summary = summaries.next()
    keys = summary.keys()
    print '%30s' % 'start_date', '\t'.join('%10s' % k for k in keys)
    print '%30s' % sd, '\t'.join('%10s' % summary[k] for k in keys)
    for sd, summary in summaries:
        print '%30s' % sd, '\t'.join('%10s' % summary[k] for k in keys)
    plotlib.plot_passages(passages)
    return 0

def handle_optimal_routes(opts, args):
    wx = list(weather(opts, args))
    if opts.ensemble:
        wx = [gfs for gfs, nww3 in wx]
    else:
        wx = [weather(opts, args).next()[0]]

    if opts.start is None and opts.end is None:
        if len(opts.waypoints) == 2:
            opts.start, opts.end = opts.waypoints
        else:
            raise ValueError("expected a single start,end for optimization")
    elif opts.start is None or opts.end is None:
        raise ValueError("expected use of either --start and --end or --waypoints")

    if opts.route_file is None:
        opts.route_file = ('%d_%d_to_%d_%d_on_%s' %
                            (opts.start.lat, opts.start.lon, opts.end.lat, opts.end.lon,
                             opts.start_date.strftime('%Y_%m_%d')))

    opts.route_file = open(opts.route_file, 'w')
    for wpts, passage in spray.iterroutes(opts.start, opts.end, opts.start_date, wx, resol=opts.resol):
        wpt_value = ':'.join(['%6.3f,%6.3f' % (x.lat, x.lon) for x in wpts])
        opts.route_file.write('%s\t%s\n' % (wpt_value, base64.b64encode(passage.dumps())))

def handle_optimal(opts, args):
    wx = list(weather(opts, args))
    if opts.ensemble:
        wx = [gfs for gfs, nww3 in wx]
    else:
        wx = [weather(opts, args).next()[0]]

    if opts.start is None and opts.end is None:
        if len(opts.waypoints) == 2:
            opts.start, opts.end = opts.waypoints
        else:
            raise ValueError("expected a single start,end for optimization")
    elif opts.start is None or opts.end is None:
        raise ValueError("expected use of either --start and --end or --waypoints")

    if opts.route_file is None:
        iterroutes = spray.iterroutes(opts.start, opts.end, opts.start_date, wx, resol=opts.resol)
    else:
        opts.route_file = open(opts.route_file, 'r')
        def decoder(x):
            wpt_str, passage_str = x.split('\t', 1)
            waypoints = [objects.LatLon(*map(float, x.split(','))) for x in wpt_str.split(':')]
            passage = core.Data(ncdf=base64.b64decode(passage_str))
            return waypoints, passage
        iterroutes = (decoder(x) for x in opts.route_file)

    passages = spray.optimal_passage(iterroutes)
    plotlib.plot_passages(passages)

def handle_plot(opts, args):
    """
    Unpacks and plots an email issued forecast
    """
    if not opts.fcst:
        raise ValueError("point to a windbreaker file with --fcst")
    f = open(opts.fcst, 'r')
    fcst = tinylib.from_beaufort(zlib.decompress(f.read()))
    mask = tinylib.rhumbline_slice(opts.waypoints[0], opts.waypoints[1], fcst['lat'].data, fcst['lon'].data)
    tinylib.tiny_slice(fcst['wind_speed'].data, )
    plotlib.plot_wind(fcst)

def handle_email(opts, args):
    """
    Processes a MIME e-mail from --input (or stdin) extracting
    a saildocs-like request and replying to the sender with
    an packed ensemble forecast.
    """
    emaillib.wind_breaker(opts.input.read(), opts.grib)

def handle_email_queue(opts, args):
    """
    Processes all MIME e-mails that have been queued up
    and which are expected to reside in --queue-directory
    """
    if not opts.queue_directory:
        raise ValueError("expected --queue_directory")
    if not os.path.isdir(opts.queue_directory):
        raise ValueError("%s is not a directory" % opts.queue_directory)
    processed_dir = os.path.join(opts.queue_directory, 'processed')
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    failed_dir = os.path.join(opts.queue_directory, 'failed')
    if not os.path.exists(failed_dir):
        os.mkdir(failed_dir)
    emails = [f for f in os.listdir(opts.queue_directory) if f.endswith('.mime')]
    logger.debug("processing %d emails from %s" % (len(emails), opts.queue_directory))
    for fn in emails:
        full_path = os.path.join(opts.queue_directory, fn)
        logger.debug("processing %s" % full_path)
        f = open(full_path, 'r')
        emaillib.wind_breaker(f.read(), opts.grib)
        f.close()
        os.rename(full_path, os.path.join(processed_dir, fn))

    logger.debug("queue processing complete.")

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
    p.add_option("", "--waypoints", default=None, action="store",
        help="lat,lon:lat,lon:lat,lon")
    p.add_option("", "--input", default=None, action="store")
    p.add_option("", "--queue-directory", default=None, action="store")
    p.add_option("", "--start-date", default=None, action="store")
    p.add_option("", "--email", default=False, action="store_true")
    p.add_option("", "--email-queue", default=False, action="store_true")
    p.add_option("", "--plot", default=False, action="store_true")
    p.add_option("", "--optimal", default=False, action="store_true")
    p.add_option("", "--optimal-routes", default=False, action="store_true")
    p.add_option("", "--simulate", default=False, action="store_true")
    p.add_option("", "--when", default=False, action="store_true")
    p.add_option("-v", "--verbose", default=False, action="store_true")
    p.add_option("", "--fast", default=False, action="store_true")
    p.add_option("", "--grib", default=None, action="store",
                 help="A grib file containing forecasts")
    p.add_option("", "--fcst", default=None, action="store",
                 help="A windbreaker file containing tiny forecasts")
    p.add_option("", "--download", default=True, action="store_true",
                 help="Download forecasts from UCAR")
    p.add_option("", "--recent", default=False, action="store_true",
                 help="Use recent forecasts from UCAR")
    p.add_option("", "--ensemble", default=False, action="store_true",
                 help="Use ensemble forecasts in route optimization")
    p.add_option("", "--resol", default=None, action="store",
                 help="The grid size for route optimization")
    p.add_option("", "--route-file", default=None, action="store",
                 help="File to store iterated optimal routes in.")

    core.ENSURE_VALID = False
    opts, args = p.parse_args()

    opts.input = open(opts.input, 'r') if opts.input else sys.stdin

    if opts.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if opts.fetch_ccmp:
        return fetch.fetch_ccmp()

    if opts.start and opts.end:
        start = objects.LatLon(*[float(x) for x in opts.start.split(',')])
        end = objects.LatLon(*[float(x) for x in opts.end.split(',')])
        opts.waypoints = [start, end]

    elif opts.waypoints:
        def degreeify(x):
            if '^' in x:
                deg, minute = map(float, x.split('^'))
            else:
                deg = float(x)
                minute = 0.
            if minute > 60:
                raise ValueError("minutes must be less than 60")
            return deg + minute/(60.)
        opts.waypoints = [objects.LatLon(*map(degreeify, x.split(',')))
                          for x in opts.waypoints.split(':')]
    else:
        logging.info("Using default start of san francisco: 36.625, -121.9")
        logging.info("Using default end of hawaii: 19.79, -154.76")
        end = objects.LatLon(19.79, -154.76) # hawaii
        start = objects.LatLon(36.625, -121.9) # sf
        opts.waypoints = [start, end]

    if opts.start_date:
        try:
            opts.start_date = datetime.datetime.strptime(opts.start_date, '%Y-%m-%d')
        except:
            opts.start_date = datetime.datetime.strptime(opts.start_date, '%Y-%m-%dT%H')
    else:
        opts.start_date = datetime.datetime.now()

    eastern = pytz.timezone('US/Pacific')
    opts.start_date = eastern.localize(opts.start_date)

    if opts.simulate:
        handle_simulate(opts, args)
    elif opts.optimal:
        handle_optimal(opts, args)
    elif opts.optimal_routes:
        handle_optimal_routes(opts, args)
    elif opts.when:
        handle_when(opts, args)
    elif opts.email:
        handle_email(opts, args)
    elif opts.email_queue:
        handle_email_queue(opts, args)
    elif opts.plot:
        handle_plot(opts, args)
    else:
        p.error("slocum completed exactly what you told it to do ... nothing.")

if __name__ == "__main__":
#    import cProfile
#    cProfile.runctx('main()', globals(), locals(),
#                    filename=os.path.join(os.path.dirname(__file__), 'profile.prof'))
    sys.exit(main())
