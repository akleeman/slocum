import argparse
import numpy as np

import matplotlib.pyplot as plt

from slocum.celestial import reduction, utils


def process_one_sight(sight):
    """
    Take each sight and apply corrections and determine the expected
    altitude observations.
    """
    lha = reduction.local_hour_angle(sight['gha'], sight['longitude'])
    observed = reduction.corrected_altitude(sight['altitude'], 0., sight['radius'])
    actual = reduction.altitude(sight['declination'], sight['latitude'], lha)
    return sight['time'], observed, actual


def noon_sight(file_name):
    """
    The main routine which ties together all the processing steps.
    """
    # read in the sights
    data = utils.read_csv(file_name)
    # Get arrays of all the times sights were made, what was
    # observed for altitude and what the actual altitude was.
    times, observed, actual = map(np.array,
                                  zip(*[process_one_sight(sight)
                                        for sight in data]))
    # Convert to integer times in order to interpolate
    dt = (times - times[0]).astype('timedelta64[ms]').astype('int')
    # interpolate
    fit = np.polyfit(dt, observed, 2)
    # find the maximum values
    max_x = -fit[1] / (2. * fit[0])
    max_dt = max_x.astype('timedelta64[ms]')
    noon_alt = np.polyval(fit, max_x)
    noon_utc = np.datetime64(times[0]) + max_dt
    # Use the sight that was nearest to noon to get the
    # declination.
    nearest_to_noon = data[np.nonzero(max_x < dt)[0][0]]
    dec = nearest_to_noon['declination']

    # then determine the two possibilities for the latitude.
    dec = np.mod(dec + 180., 360.) - 180.
    lower_noon_lat = noon_alt - 90. + dec
    upper_noon_lat = 90. - noon_alt + dec    

    # If we gave an apriori latitude we can use that to choose the
    # closest latitude, otherwise we just print them both.
    if np.isfinite(nearest_to_noon['latitude']):
        if (np.abs(nearest_to_noon['latitude'] - lower_noon_lat) <
            np.abs(nearest_to_noon['latitude'] - upper_noon_lat)):
            noon_lat = lower_noon_lat
        else:
            noon_lat = upper_noon_lat
        print ("Your latitude at %s was %s" %
               (noon_utc, utils.decimal_to_degrees_minutes(noon_lat)))
    else:
        print ("Your latitude at %s was either %s or %s"
               % (noon_utc,
                  utils.decimal_to_degrees_minutes(lower_noon_lat),
                  utils.decimal_to_degrees_minutes(upper_noon_lat)))

    # Create a plot showing the observed and expected time series
    # of altitudes.
    interp_xs = np.linspace(np.minimum(0., max_x),
                            np.max(dt), 1000)
    interp_vals = np.polyval(fit, interp_xs)
    interp_tds = interp_xs.astype('timedelta64[ms]')
    interp_times = times[0] + interp_tds
    interp_actual = np.polyval(np.polyfit(dt, actual, 2), interp_xs)

    plt.plot(interp_times, interp_vals)
    plt.plot(interp_times, interp_actual)
    plt.plot(times.astype('datetime64[ms]'), observed, 'k.')
    plt.plot(times[0] + max_x.astype('timedelta64[ms]'), noon_alt, 'r.')
    plt.legend(['polynomial_observed', 'expected'])
    plt.show()


def main(args):
    noon_sight(args.input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    noonsight.py

    A tool to determine your latitude from a series of noon sights.""")
    parser.add_argument('input',
                        help=("path to a csv file holding sights"))
    args = parser.parse_args()
    main(args)
