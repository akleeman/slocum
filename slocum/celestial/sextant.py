import argparse
import numpy as np
import matplotlib.pyplot as plt

from slocum.celestial import reduction, utils, visualize


def main(args):

    if args.input is None:
        sight = {'time': args.utc,
                 'gha': args.gha,
                 'declination': args.dec,
                 'latitude': args.latitude,
                 'longitude': args.longitude,
                 'altitude': args.altitude,
                 'radius': 16.1}
        sights = [utils.parse_one_sight(sight)]
    else:
        sights = utils.read_csv(args.input)

    def get_lop(sight):
        lon, lat, z = reduction.line_of_position(sight.get('time'),
                                                 sight.get('declination'),
                                                 sight.get('gha'),
                                                 sight.get('longitude'),
                                                 sight.get('latitude'),
                                                 sight.get('altitude'))
        print ("You are likely on the line passing through %s, %s"
               "at an angle of %.0f from north."
               % (utils.decimal_to_degrees_minutes(lat),
                  utils.decimal_to_degrees_minutes(lon),
                  np.mod(z + 90. + 180., 360.) - 180.))
        return lon, lat, z
    
    lons, lats, zs = zip(*[get_lop(sight) for sight in sights])

    actual_lons = [sight['longitude'] for sight in sights]
    all_lons = np.concatenate([lons, actual_lons])

    actual_lats = [sight['latitude'] for sight in sights]
    all_lats = np.concatenate([lats, actual_lats])

    bm = visualize.get_basemap(all_lons, all_lats, pad=2.)

    for lon, lat, z in zip(lons, lats, zs):
        bm = visualize.plot_line_of_position(lon, lat, z, bm=bm)
    
    for sight in sights:
        bm.scatter(*bm(sight['longitude'], sight['latitude']))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    sextant.py

    A tool to help perform sight reductions.""")
    parser.add_argument("input", default=None,
                        help="A path to a csv file holding sights")
    parser.add_argument("--altitude",
                        help="The sighted altitude of the body.")
    parser.add_argument("--utc",
                        help="The time the body was observed.")
    parser.add_argument("--latitude",
                        help="The approximate latitude")
    parser.add_argument("--longitude",
                        help="The approximate longitude")
    parser.add_argument("--gha",
                        help=("The Greenwich hour angle.  Can either"
                              " be a single value or a pair of values"
                              " ,corresponding to start/end of the hour,"
                              " in which case the values will be interpolated"))
    parser.add_argument("--dec",
                        help=("The declination angle.  Can either"
                              " be a single value or a pair of values"
                              " ,corresponding to start/end of the hour,"
                              " in which case the value will be interpolated"))
    args = parser.parse_args()
    main(args)
