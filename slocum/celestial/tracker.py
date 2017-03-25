import pyproj
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg
from matplotlib import patches

from slocum.lib import units
from slocum.celestial import reduction, utils, visualize

SIG_SOG = 0.5
SIG_COG = np.deg2rad(15.)
SIG_INIT = 10.


def build_state(positions_with_cov):
    mu = np.zeros(2 * len(positions_with_cov))
    covs = np.cumsum([p['covariance'] for p in positions_with_cov], axis=0)
    cov = linalg.block_diag(*covs)
    references = np.concatenate([[p['longitude'], p['latitude']]
                                 for p in positions_with_cov])

    return mu, cov, references


def next_or_none(iterator):
    try:
        next_val = iterator.next()
    except StopIteration:
        next_val = None
    return next_val


def dead_reakon_covariance(distance, azimuth, sig_d, sig_a):
    theta = np.deg2rad(azimuth)
    J = np.array([[np.sin(theta), distance * np.cos(theta)],
                  [np.cos(theta), -distance * np.sin(theta)]])
    return reduce(np.dot, [J, np.diag([sig_d ** 2, sig_a ** 2]), J.T])


def dead_reakon(courses, times):
    prev_lon = courses[0]['longitude']
    prev_lat = courses[0]['latitude']

    iter_courses = iter(courses)
    course = iter_courses.next()
    next_course = next_or_none(iter_courses)

    geod = pyproj.Geod(ellps="sphere")

    prev_time = times[0]

    for time in times:
        while next_course is not None and time > next_course['time']:
            course = next_course
            next_course = next_or_none(iter_courses)
        
        dt_sec = (time - prev_time).astype('timedelta64[s]')

        sog_ms = units.convert_scalar(course['sog'], 'knots', 'm/s')

        distance = sog_ms * dt_sec.astype('int') / 1000.
        lon, lat, _ = geod.fwd(prev_lon, prev_lat, course['cog'], distance * 1000)

        sig_cog = course.get('sig_cog', SIG_COG)
        sig_sog = course.get('sig_sog', SIG_SOG)

        sig_d = sig_sog * sog_ms * dt_sec.astype('int') / 1000.

        if time == times[0]:
            cov = SIG_INIT ** 2 * np.eye(2)
        else:
            cov = dead_reakon_covariance(distance, course['cog'], sig_d, sig_cog)


        yield {'longitude': lon,
               'latitude': lat,
               'covariance': cov}
        prev_lon = lon
        prev_lat = lat
        prev_time = time


def plot_estimate(bm, mu, cov, ref, fig, ax):
    lon_lats = ref.reshape(ref.size / 2, 2)
    geod = pyproj.Geod(ellps="sphere")

    for i in range(ref.size / 2):
        sl = slice(2 * i, 2 * i + 2)
        lon, lat = ref[sl]
        one_cov = cov[sl, sl]

        (w, h), eig_vecs = np.linalg.eig(one_cov)

        w = np.sqrt(w)
        h = np.sqrt(h)

        angle = np.rad2deg(np.arctan2(*np.dot(eig_vecs, [1., 1.])))

        lons, lats, _ = geod.fwd([lon, lon], [lat, lat],
                                 [angle, angle + 90.],
                                 [w * 1000, h * 1000])

        x_center, y_center = bm(lon, lat)
        x_axes, y_axes = bm(lons, lats)

        dists = np.sqrt(np.square(x_axes - x_center) + np.square(y_axes - y_center))

        ellipse = patches.Ellipse(bm(lon, lat), dists[0], dists[1], angle=angle,
                                  edgecolor='k', facecolor='w',
                                  zorder=100, alpha=0.1)
        ax.add_patch(ellipse)


def main(args):
    sights = utils.read_sights(args.sights)
    courses = utils.read_courses(args.courses)

    times = itertools.chain([s['time'] for s in sights],
                            [c['time'] for c in courses])

    times = sorted(set(times))

    apriori_positions = list(dead_reakon(courses, times))

    mu, cov, ref = build_state(apriori_positions)


    lons = np.array([s['longitude'] for s in sights])
    lats = np.array([s['latitude'] for s in sights])

    fig, ax = plt.subplots(1, 1)
    bm = visualize.get_basemap(lons, lats, ax=ax)
    bm.plot(lons, lats, 'k')
    bm.scatter(lons, lats, color='k')

    plot_estimate(bm, mu, cov, ref, fig, ax)
    plt.show()
    import ipdb; ipdb.set_trace()




    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    tracker.py

    A tool to determine your latitude from a series of sights and DR.""")
    parser.add_argument('sights',
                        help=("path to a csv file holding sights."))
    parser.add_argument('courses',
                        help=("path to a csv of course changes."))
    args = parser.parse_args()
    main(args)
