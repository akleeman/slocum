import pyproj
import argparse
import numpy as np

import matplotlib.pyplot as plt

from slocum.lib import units
from slocum.celestial import reduction, utils, visualize


def noon_sight(file_name):
    """
    The main routine which ties together all the processing steps.
    """
    # read in the sights
    data = utils.read_csv(file_name)

    def get_lop(sight):
        return reduction.line_of_position(sight['time'],
                                          sight['declination'],
                                          sight['gha'],
                                          sight['longitude'],
                                          sight['latitude'],
                                          sight['altitude'],
                                          radius=sight['radius'])

    lops = map(get_lop, data)
    lons, lats, zs = map(np.array, zip(*lops))

    geod = pyproj.Geod(ellps="sphere")
    def make_line(lop):
        lon, lat, z = lop
        def line(x):
            dist_m = units.convert_scalar(x, 'nautical_mile', 'm')
            new_lon, new_lat, _ = geod.fwd(lon, lat, z, dist_m)
            return new_lon, new_lat
        return line
    lines = map(make_line, lops)

    def make_equations(lop):
        lon, lat, z = lop
        sin_z = np.sin(np.deg2rad(z))
        return np.array([1., -sin_z]), lat -sin_z * lon
    
    def other_line(lop):
        lon, lat, z = lop
        print z
        def line(x):
            new_lat = lat + x / 60.
            new_lon = lon + x / 60. / (np.cos(np.deg2rad(lat)) * np.tan(np.deg2rad(z)))
            return new_lon, new_lat
        return line
    other_lines = map(other_line, lops)
    
    A, b = map(np.array, zip(*map(make_equations, lops)))
    lat, lon = np.linalg.solve(A, b)
#     y - sin(z0) x = lat0 - sin(z0) lon0
#     y - sin(z1) x = lat1 - sin(z1) lon1

    bm = visualize.get_basemap(lons, lats, pad=1.)
    deltas = np.linspace(-10., 10., 101)

    for line in lines:
        line_lons, line_lats = zip(*[line(x) for x in deltas])
        xs, ys = bm(line_lons, line_lats)
        bm.plot(xs, ys)
#
#     for line in other_lines:
#         line_lons, line_lats = zip(*[line(x) for x in deltas])
#         xs, ys = bm(line_lons, line_lats)
#         bm.plot(xs, ys)

    bm.scatter(data[0]['longitude'], data[0]['latitude'], s=20, color='red')
    bm.scatter(lon, lat, s=20, color='black')

    plt.show()
    import ipdb; ipdb.set_trace()


def main(args):
    noon_sight(args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    multibody.py

    A tool to determine your position from a pair of simultaneous
    (or near simultaneous) sights of two different celestial bodies.""")
    parser.add_argument('input',
                        help=("path to a csv file holding sights"))
    args = parser.parse_args()
    main(args)
