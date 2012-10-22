import os
import sys
import datetime

from optparse import OptionParser

def main():
    p = OptionParser(usage="""%%prog [options]

    Queues up an email for processing with slocum.py --email

    """)
    p.add_option("", "--directory", default=None, action="store")
    opts, args = p.parse_args()
    if not opts.directory:
        p.error("--directory option is required")
    if not os.path.isdir(opts.directory):
        raise ValueError("--directory should point to an existing directory")

    filename = 'wind_breaker_query_%s.mime' % datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    f = open(os.path.join(opts.directory, filename), 'w')
    f.write(sys.stdin.read())
    f.close()

if __name__ == "__main__":
    sys.exit(main())