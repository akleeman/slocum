#!/usr/bin/python2.6
"""
gribapi package
https://software.ecmwf.int/wiki/display/GRIB/Releases

./configure --prefix=/export/disk0/wb/python2.6/ --enable-python
make
make install
echo grib_api > /export/disk0/wb/python2.6/lib/python2.7/site-packages/gribapi.pth

"""
import os
import sys
import zlib
import numpy as np
import logging
import argparse

from optparse import OptionParser
from matplotlib import pyplot as plt

import sl.objects.conventions as conv

from sl import poseidon
from sl.lib import emaillib, tinylib
from sl.objects import objects
from polyglot.data import Dataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

def handle_email(args):
    """
    Processes a MIME e-mail from --input (or stdin) extracting
    a saildocs-like request and replying to the sender with
    an packed ensemble forecast.
    """
    opts.output = open(opts.output, 'w') if opts.output else None
    emaillib.windbreaker(opts.input.read(), opts.grib, opts.output)

def handle_email_queue(args):
    """
    Processes all MIME e-mails that have been queued up
    and which are expected to reside in --queue-directory
    """
    if not opts.queue_directory:
        raise ValueError("expected --queue_directory")
    opts.output = open(opts.output, 'w') if opts.output else None
    emaillib.windbreaker_queue(opts.queue_directory,
                               ncdf_weather=opts.grib,
                               catchable_exceptions=opts.exceptions,
                               output=opts.output)

def add_coordinate(obj, name, vars):

    data = np.asarray([0])

    obj.create_coordinate(name=name,
                          data=data)

    for i, vname in enumerate(vars):
        var = obj[vname]
        newshape = tuple([1] + list(var.data.shape))
        expanded_dim = tuple([name] + list(var.dimensions))
        vdata = var.data.reshape(newshape)
        obj.delete_variable(vname)
        obj.create_variable(name=vname,
                            dims=expanded_dim,
                            attributes=var.attributes,
                            data=vdata)

def handle_test(args):
    ll = objects.LatLon(-45, 160)
    ur = objects.LatLon(-35, 180)
    temp_file = '/home/kleeman/Desktop/gefs_test.nc'
    if not os.path.exists(temp_file):
        gefs = poseidon.gefs(ll, ur)
        gefs.dump(temp_file)

    import copy
    gefs = copy.deepcopy(Dataset(temp_file))

    from sl.objects import units
    units.normalize_units(gefs[conv.UWND])
    units.normalize_units(gefs[conv.VWND])

    import zlib
    tiny = zlib.compress(tinylib.to_beaufort(gefs))
    print len(tiny) / float(os.path.getsize(temp_file))
    full = tinylib.from_beaufort(zlib.decompress(tiny))

    from sl.lib import griblib
    griblib.save(gefs, 'test.grb')


_task_handler = {'email': handle_email,
                 'test': handle_test}

if __name__ == "__main__":

    parser = argparse.ArgumentParser("""%(prog)s task [options]""",
                                     description = """
    Slocum -- A tool for ocean passage planning

    Joshua Slocum (February 20, 1844 -on or shortly after November 14, 1909)
    was a Canadian-American seaman and adventurer, a noted writer, and the first
    man to sail single-handedly around the world. In 1900 he told the story of
    this in Sailing Alone Around the World. He disappeared in November 1909
    while aboard his boat, the Spray. (wikipedia)""")

    # add subparser for each task
    subparsers = parser.add_subparsers()
    def add_parser(k):
        p = subparsers.add_parser(k)
        p.set_defaults(func=_task_handler[k])
        p.add_argument('--input', type=argparse.FileType('r'),
                        default=sys.stdin)
        p.add_argument('--output', type=argparse.FileType('w'),
                            default=sys.stdout)
        return p
    task_parsers = [add_parser(k) for k in _task_handler.keys()]
    # parse the arguments and run the handler associated with each task
    args = parser.parse_args()
    args.func(args)
