#!/usr/bin/python2.7

import os
import sys
import logging
import argparse

# Configure the logger
fmt = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
logging.basicConfig(filename='/tmp/slocum.log',
                    level=logging.DEBUG,
                    format=fmt)
logger = logging.getLogger(os.path.basename(__file__))
file_handler = logging.FileHandler("/tmp/slocum.log")
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

from sl.lib import emaillib


def handle_email(args):
    """
    Processes a MIME e-mail from --input (or stdin) extracting
    a saildocs-like request and replying to the sender with
    an packed ensemble forecast.
    """
    emaillib.windbreaker(args.input.read(), None, args.output)

_task_handler = {'email': handle_email}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    Slocum -- A tool for ocean passage planning

    Joshua Slocum (February 20, 1844 -on or shortly after November 14, 1909)
    was a Canadian-American seaman and adventurer, a noted writer, and the
    first man to sail single-handedly around the world. In 1900 he told the
    story of this in Sailing Alone Around the World. He disappeared in
    November 1909 while aboard his boat, the Spray. (wikipedia)""")

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
