import os
import sys
import zlib
import logging
import argparse
import tempfile
import datetime as dt

# Configure the logger
fmt = "%(asctime)s [%(filename)-12.12s] [%(levelname)-5.5s]  %(message)s"
logging.basicConfig(filename='/tmp/slocum.log',
                    level=logging.DEBUG,
                    format=fmt)

logger = logging.getLogger(os.path.basename(__file__))
file_handler = logging.FileHandler("/%s/slocum.log" % tempfile.gettempdir())
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stderr)
logger.addHandler(console_handler)
logger.setLevel("INFO")

import windbreaker
from lib import (griblib, tinylib, rtefcst, enslib, saildocs, conventions)


def handle_spot(args):
    """
    Converts a packed spot forecast to a spot text message.
    """
    from lib import visualize
    payload = args.input.read()
    fcsts = tinylib.from_beaufort(payload)
    if conventions.ENSEMBLE in fcsts:
        assert fcsts[conventions.LAT].size == 1
        assert fcsts[conventions.LON].size == 1
        fcsts = fcsts.isel(**{conventions.LON: 0, conventions.LAT: 0})
        visualize.spot_plot(fcsts)
    else:
        windbreaker.spot_message(fcsts, args.output)


def handle_netcdf(args):
    """
    Converts a packed ensemble forecast to a netCDF4 file.
    """
    tinyfcst = zlib.decompress(args.input.read())
    fcst = tinylib.from_beaufort(tinyfcst)
    out_file = args.output.name
    args.output.close()
    fcst.dump(out_file)


def handle_grib(args):
    """
    Converts a packed ensemble forecast to a standard GRIB.
    """
    tinyfcst = zlib.decompress(args.input.read())
    fcst = tinylib.from_beaufort(tinyfcst)
    griblib.save(fcst, target=args.output, append=False)


def handle_query(args):
    """
    Process a queries from the command line.  This is mostly used
    for debuging.
    """
    queries = list(saildocs.iterate_query_strings(args.input.read()))
    if len(queries) != 1:
        raise NotImplementedError("Can only process one query at a time")
    query = windbreaker.parse_query(queries.pop(0))
    args.output.write(windbreaker.query_to_beaufort(query,
                                                    args.forecast))


def handle_email(args):
    """
    Processes a MIME e-mail from --input (or stdin) extracting
    a saildocs-like request and replying to the sender with
    an packed ensemble forecast.
    """
    try:
        # process the email
        windbreaker.process_email(args.input.read(), args.forecast,
                                output=args.output,
                                fail_hard=args.fail_hard,
                                log_input=True)
    except Exception, e:
        logging.exception(e)
        raise


def handle_route_forecast(args):
    """
    Generates a gpx waypoint file with wind forecast info along a route
    provided in an input file.
    """
    tinyfcst = zlib.decompress(args.input.read())
    args.input.close()
    fcst = tinylib.from_beaufort(tinyfcst)

    # TODO: replace datetime by np.datetime64 and allow local time or utc
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
    p.add_argument('--input', type=argparse.FileType('rb'),
                   default=sys.stdin)
    p.add_argument('--output', type=argparse.FileType('wb'),
                   default=sys.stdout)
    p.add_argument('--forecast', default=None,
                   help="path to a netCDF forecast")
    p.add_argument('--fail-hard', default=False,
                   action='store_true')


def setup_parser_route_forecast(p):
    """
    Configures the argument subparser for handle_route_forecast.  p is the
    ArgumentParser object for the route_forecast subparser.
    """
    p.add_argument('--input', metavar='fcst_file',
                   type=argparse.FileType('rb'), default=sys.stdin,
                   help='windbreaker forecast file; stdin if not specified')
    p.add_argument('--output', metavar='gpx_file', type=argparse.FileType('w'),
                   help=('waypoints with wind data along route; will '
                         'overwrite if exists; stdout if not specified'),
                   default=sys.stdout)
    p.add_argument('--rtefile', metavar='file',
                   type=argparse.FileType('r'), required=True,
                   help='input file with route definition')
    p.add_argument('--rtefmt', metavar='fmt', choices=['csv', 'gpx'],
            required=True, help='format of route input file; valid formats: '
            'csv, gpx')
    utNowStr = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M')
    p.add_argument('--utcdept', metavar='YYYY-mm-ddTHH:MM', default=utNowStr,
                   help='utc of departure; default is current utc')
    p.add_argument('--speed', metavar='SOG', type=float,
                   help=('expected average speed over ground in kn; can '
                         'be ommitted if rtefile contains speeds for each '
                         'leg of the route'))
    p.add_argument('--truewind', action='store_true',
                   help=('if specified, output will show true rather than '
                         'apparent wind at forecast waypoints'))
    p.add_argument('--notimelabel', action='store_true',
                   help=('if specified, time labels will be ommitted from '
                         'forecast waypoint names'))


def setup_parser_spot(p):

    variable_choices = [fv[0] for fv in enslib._fcst_vars]
    p.add_argument(
            '--input', metavar='FILE', required='True',
            type=argparse.FileType('rb'), help="input file with "
            "windbreaker SPOT forecast ensemble")
    p.add_argument(
            '--variable', metavar='VARIABLE', choices=variable_choices,
            help="forecast variable for which to create plot; valid "
            "choices: %s; combined plot will be created if not specified" %
            ', '.join(variable_choices))
    p.add_argument(
            '--plot', metavar='TYPE', choices=['box', 'bar'],
            default='box', help="plot type to be created ('bar'|'box'), "
            "defaults to 'box'; ignored if no forecast variable is specified "
            "in which case boxplots for wind and pressure will be created")
    p.add_argument(
            '--export', metavar='PATH', help="if specified, the plot "
            "will be saved to the directory specified by PATH under the "
            "name 'se_<lat-lon>_<t0>_<plot type>.svg'")


# The _task_handler dictionary maps each 'command' to a (task_handler,
# parser_setup_handler) tuple.  Subparsers are initialized in __main__  (with
# the handler function's doc string as help text) and then the appropriate
# setup handler is called to add the details.
_task_handler = {'email': (handle_email, setup_parser_email),
                 'query': (handle_query, setup_parser_email),
                 'grib': (handle_grib, setup_parser_grib),
                 'netcdf': (handle_netcdf, setup_parser_grib),
                 'route-forecast': (handle_route_forecast,
                                    setup_parser_route_forecast),
                 'spot': (handle_spot, setup_parser_spot)}


def main():
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

if __name__ == "__main__":
    main()
