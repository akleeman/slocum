import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

from slocum.celestial import reduction, utils


def main(args):
    sights = list(itertools.chain(*[utils.read_csv(file_name)
                                    for file_name in args.inputs]))

    
    def expected(sight):
        lha = reduction.local_hour_angle(sight['gha'],
                                         sight['longitude'])
        return reduction.altitude(sight['declination'],
                                  sight['latitude'],
                                  lha)
    expected_alts = np.array(map(expected, sights))
        
    def actual(sight, height_m=0.):
        return reduction.corrected_altitude(sight['altitude'],
                                            height_m,
                                            sight['radius'])
    actual_alts = np.array(map(actual, sights))

#     def mse(height_m):
#         actual_alts = np.array([actual(sight, height_m)
#                                 for sight in sights])
#         return np.mean(np.square(actual_alts - expected_alts))
#
#     ret = optimize.fminbound(mse, 0., 10.)
#
#     xs = np.linspace(0., 10., 101)
#     mses = np.array([mse(x) for x in xs])
#     plt.plot(xs, mses)
#     plt.show()
#
#     import ipdb; ipdb.set_trace()

    plt.plot(expected_alts, actual_alts, 'k.')
    plt.plot([np.min(expected_alts), np.max(expected_alts)],
             [np.min(expected_alts), np.max(expected_alts)],
             'b:')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    sextant.py

    A tool to help perform sight reductions.""")
    parser.add_argument("inputs", default=None, nargs="*",
                        help="A path to csv files holding sights")
    args = parser.parse_args()
    main(args)
