import os

from slocum.query import saildocs

if __name__ == "__main__":
    # Builds the gridded query usage file.
    curdir = os.path.dirname(__file__)
    with open(os.path.join(curdir, 'gridded_usage.rst'), 'w') as f:
        f.write(saildocs._gridded_usage)

    # Builds the gridded query usage file.
    curdir = os.path.dirname(__file__)
    with open(os.path.join(curdir, 'spot_usage.rst'), 'w') as f:
        f.write(saildocs._spot_usage)
