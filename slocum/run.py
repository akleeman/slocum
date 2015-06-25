import os
import sys
import zlib
import logging
import argparse
import tempfile
import datetime as dt

# Configure the logger
fmt = "%(asctime)s [%(filename)-12.12s] [%(levelname)-5.5s]  %(message)s"
_log_path = os.path.join(tempfile.gettempdir(), "slocum.log")
logging.basicConfig(filename=_log_path,
                    level=logging.DEBUG,
                    format=fmt)

logger = logging.getLogger(os.path.basename(__file__))
file_handler = logging.FileHandler(_log_path)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stderr)
logger.addHandler(console_handler)
logger.setLevel("INFO")

from query import request
from compression import compress


def handle_plot(args):
    # we save these imports so the script can be run as a server
    # without requiring matplotlib.
    import matplotlib.pyplot as plt
    import visualize

    # read in the input
    if args.input is None:
        raise argparse.ArgumentError(args.input,
                                     "--input is required, specify -h for usage.")
    payload = args.input.read()
    # decompress
    fcst = compress.decompress_dataset(payload)
    # plot
    visualize.plot_forecast(fcst)
    # save or show
    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output.name)


def handle_query(args):
    """
    Process a queries from the command line.  This is mostly used
    for debugging.
    """
    queries = list(request.iterate_query_strings(args.input.read()))
    if len(queries) != 1:
        raise NotImplementedError("Can only process one query at a time")
    query = request.parse_query_string(queries.pop(0))
    args.output.write(request.process_query(query,
                                            args.forecast))


def handle_email(args):
    """
    Processes a MIME e-mail from --input (or stdin) extracting
    a saildocs-like request and replying to the sender with
    an packed ensemble forecast.
    """
    try:
        args.input = args.input or sys.stdin
        # process the email
        request.process_email(args.input.read(),
                              url=args.forecast,
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
    # TODO
    raise NotImplementedError("This needs to be refactored to work "
                              "with the new version of slocum")
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


def add_common_arguments(p):
    # This allows the user to explicitly specify the input
    # argument using --input, or infer it as the first
    # positional argument.
    p.add_argument('input_file', metavar='input', nargs='?',
                   type=argparse.FileType('rb'))
    p.add_argument('--input',
                   default=sys.stdin,
                   type=argparse.FileType('rb'))
    # similarly, output is either explicitly specified using
    # --output, or inferred as the second positional argument.
    p.add_argument('output_file', metavar='output', nargs='?',
                   type=argparse.FileType('wb'))
    p.add_argument('--output',
                   default=sys.stdout,
                   type=argparse.FileType('wb'))
    p.add_argument('--profile', default=None)


def setup_parser_email(p):
    """
    Configures the argument subparser for handle_email.  p is the
    ArgumentParser object for the route_forecast subparser.
    """
    add_common_arguments(p)
    p.add_argument('--forecast', default=None,
                   help="path to a netCDF forecast")
    p.add_argument('--cache-forecast',
                   default=None,
                   help="path where subset forecast is cached.")
    p.add_argument('--fail-hard', default=False,
                   action='store_true')


def setup_parser_route_forecast(p):
    """
    Configures the argument subparser for handle_route_forecast.  p is the
    ArgumentParser object for the route_forecast subparser.
    """
    add_common_arguments(p)
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
                         'be omitted if rtefile contains speeds for each '
                         'leg of the route'))
    p.add_argument('--truewind', action='store_true',
                   help=('if specified, output will show true rather than '
                         'apparent wind at forecast waypoints'))
    p.add_argument('--notimelabel', action='store_true',
                   help=('if specified, time labels will be omitted from '
                         'forecast waypoint names'))


# The _task_handler dictionary maps each 'command' to a (task_handler,
# parser_setup_handler) tuple.  Subparsers are initialized in __main__  (with
# the handler function's doc string as help text) and then the appropriate
# setup handler is called to add the details.
_task_handler = {'email': (handle_email, setup_parser_email),
                 'query': (handle_query, setup_parser_email),
                 'route-forecast': (handle_route_forecast,
                                    setup_parser_route_forecast),
                 'plot': (handle_plot, add_common_arguments),}


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
    args.input = args.input_file or args.input
    args.output = args.output_file or args.output
    if args.profile:
        import cProfile
        cProfile.runctx('args.func(args)', globals(), locals(), args.profile)
    else:
        args.func(args)

if __name__ == "__main__":
    main()
