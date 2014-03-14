#!/usr/bin/python2.7
import os
import sys
import zlib
import logging
import argparse
import datetime as dt

# Configure the logger
fmt = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
logging.basicConfig(filename='/tmp/slocum.log',
                    level=logging.DEBUG,
                    format=fmt)

logger = logging.getLogger(os.path.basename(__file__))
file_handler = logging.FileHandler("/tmp/slocum.log")
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)

from sl import windbreaker
from sl.lib import griblib, tinylib, rtefcst


def handle_grib(args):
    """
    Converts a packed ensemble forecast to a standard GRIB
    """
    tinyfcst = zlib.decompress(args.input.read())
    fcst = tinylib.from_beaufort(tinyfcst)
    griblib.save(fcst, target=args.output, append=False)


def handle_email(args):
    """
    Processes a MIME e-mail from --input (or stdin) extracting
    a saildocs-like request and replying to the sender with
    an packed ensemble forecast.
    """
    windbreaker.windbreaker(args.input.read(), args.ncdf, output=args.output)


def handle_route_forecast(args):
    """
    Generates a gpx waypoint file with wind forecast info along a route
    provided in an input file.
    """
#     tinyfcst = zlib.decompress(args.input.read())
#     args.input.close()
#     fcst = tinylib.from_beaufort(tinyfcst)

    import xray
    fcst = xray.open_dataset(args.input.name)

    if args.utcdept:
        ut = dt.datetime.strptime(args.utcdept, '%Y-%m-%dT%H:%M')
    else:
        ut = None
    rte = rtefcst.Route(ifh=args.rtefile, inFmt=args.rtefmt, utcDept=ut,
                        avrgSpeed=args.speed)
    args.rtefile.close()

    rf = rtefcst.RouteForecast(rte, fcst)
    if args.truewind:
        windApparent = False
    else:
        windApparent = True
    if args.notimelabel:
        timeLabels = False
    else:
        timeLabels = True
    rf.exportFcstGPX(args.output, windApparent, timeLabels)
    args.output.close()


def setup_parser_grib(p):
    """
    Configures the argument subparser for handle_grib.  p is the
    ArgumentParser object for the route_forecast subparser.
    """
    p.add_argument('--input', type=argparse.FileType('rb'), default=sys.stdin)
    p.add_argument('--output', type=argparse.FileType('wb'),
                   default=sys.stdout)


def setup_parser_email(p):
    """
    Configures the argument subparser for handle_email.  p is the
    ArgumentParser object for the route_forecast subparser.
    """
    p.add_argument('--input', type=argparse.FileType('rb'), default=sys.stdin)
    p.add_argument('--output', type=argparse.FileType('wb'),
                   default=sys.stdout)
    p.add_argument('--ncdf', default=None)


def setup_parser_route_forecast(p):
    """
    Configures the argument subparser for handle_route_forecast.  p is the
    ArgumentParser object for the route_forecast subparser.
    """
    p.add_argument('--input', metavar='fcst_file',
                   type=argparse.FileType('rb'), default=sys.stdin,
                   help='windbreaker forecast file; stdin if not specified')
    p.add_argument('--output', metavar='gpx_file', type=argparse.FileType('w'),
                   help=('waypoints with wind data along route; will ' +
                         'overwrite if exists; stdout if not specified'),
                   default=sys.stdout)
    p.add_argument('--rtefile', metavar='file',
                   type=argparse.FileType('r'), required=True,
                   help='input file with route definition')
    p.add_argument('--rtefmt', metavar='fmt', choices=['csv'], default='csv',
                   help='format of route input file; valid formats: csv')
    utNowStr = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M')
    p.add_argument('--utcdept', metavar='YYYY-mm-ddTHH:MM', default=utNowStr,
                   help='utc of departure; default is current utc')
    p.add_argument('--speed', metavar='SOG', type=float,
                   help=('expected average speed over ground in kn; can ' +
                         'be ommitted if rtefile contains speeds for each ' +
                         'leg of the route'))
    p.add_argument('--truewind', action='store_true',
                   help=('if specified, output will show true rather than ' +
                         'apparent wind at forecast waypoints'))
    p.add_argument('--notimelabel', action='store_true',
                   help=('if specified, time labels will be ommitted from ' +
                         'forecast waypoint names'))


# The _task_handler dictionary maps each 'command' to a (task_handler,
# parser_setup_handler) tuple.  Subparsers are initialized in __main__  (with
# the handler function's doc string as help text) and then the appropriate
# setup handler is called to add the details.
_task_handler = {'email': (handle_email, setup_parser_email),
                 'grib': (handle_grib, setup_parser_grib),
                 'route-forecast': (handle_route_forecast,
                                    setup_parser_route_forecast)}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    Slocum -- A tool for ocean passage planning.

    Joshua Slocum (February 20, 1844 -on or shortly after November 14, 1909)
    was a Canadian-American seaman and adventurer, a noted writer, and the
    first man to sail single-handedly around the world. In 1900 he told the
    story of this in Sailing Alone Around the World. He disappeared in
    November 1909 while aboard his boat, the Spray. (wikipedia)""")

    # add subparser for each task
    subparsers = parser.add_subparsers()

    for k in _task_handler.keys():
        func, p_setup = _task_handler[k]
        p = subparsers.add_parser(k, help=func.__doc__)
        p.set_defaults(func=func)
        p_setup(p)

    # parse the arguments and run the handler associated with each task
    args = parser.parse_args()
    args.func(args)
