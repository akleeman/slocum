import ephem
import pyproj
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from scipy import linalg
from matplotlib import patches

from slocum.lib import units
from slocum.celestial import reduction, utils, visualize
from xml.etree.ElementInclude import include

SIG_SOG = 0.5
SIG_COG = np.deg2rad(10.)
SIG_INIT = 2.
SIG_ALT = units.convert_scalar(5., 'nautical_mile', 'm') / 1000.

_bodies = {'sun': ephem.Sun(),
           'venus': ephem.Venus()}


def build_state(positions_with_cov):
    mu = np.zeros(2 * len(positions_with_cov))
    covs = [p['covariance'] for p in positions_with_cov]

    cov = np.zeros((mu.size, mu.size))
    for i, cc in enumerate(covs):
        # This fills in the lower right portions with replications
        # of the current covariance.
        cov[2 * i:, 2 * i:] += np.kron(np.ones((mu.size / 2 - i,
                                                mu.size / 2 - i)),
                                      cc)

    references = np.concatenate([[p['longitude'], p['latitude']]
                                 for p in positions_with_cov])

    return mu, cov, references


def next_or_none(iterator):
    try:
        next_val = iterator.next()
    except StopIteration:
        next_val = None
    return next_val


def kalman_update(x, P, y, H, R):
    # actual observation minus the observation model
    innov = y - np.dot(H, x)
    PHT = np.dot(P, H.T)
    S = np.dot(H, PHT) + R
    x = x + PHT.dot(np.linalg.solve(S, innov))
    # The Kalman Gain K is
    #   K = P_{k|k-1} H^T S^{-1}
    # The Kalman Update in Joseph form is
    #   P_{k|k} = (I - KH) P_{k|k-1} (I - KH)^T + K R K^T
    # Which reduces to:
    #   P_{k|k} = (I - KH) P_{k|k-1}
    # when using the optimal Kalman Gain.  The Joseph gain
    # formulation is thought to be more numerically stable,
    # and guarentees symmetry. Though it still differences (I - KH)
    # so doesn't guarentee numerical positive definiteness.
    # See: https://en.wikipedia.org/wiki/Kalman_filter#Deriving_the_a_posteriori_estimate_covariance_matrix
    # and Chapter 10 of Gibbs Advanced Kalman Filtering.
    IKH = np.eye(P.shape[0]) - PHT.dot(np.linalg.solve(S, H))
    KsqrtR = PHT.dot(np.linalg.solve(S, np.linalg.cholesky(R)))
    P = reduce(np.dot, [IKH, P, IKH.T]) + np.dot(KsqrtR, KsqrtR.T)
    # Compute the residual corresponding to this estimate.
    resid = y - np.dot(H, x)
    # Approximate for speed: don't re-linearize h(x) before computing
    # the posterior covariance in the measurement space.
    S = reduce(np.dot, [H, P, H.T]) + R
    return x, P, resid, S


def dead_reakon_covariance(distance, azimuth, sig_d, sig_a):
    theta = np.deg2rad(azimuth)
    J = np.array([[np.sin(theta), distance * np.cos(theta)],
                  [np.cos(theta), -distance * np.sin(theta)]])
    return reduce(np.dot, [J, np.diag([sig_d ** 2, sig_a ** 2]), J.T])


def dead_reakon(courses, times):
    prev_lon = courses[0]['longitude']
    prev_lat = courses[0]['latitude']

    sig_init = courses[0]['sigma']
    if np.isnan(sig_init):
        sig_init = SIG_INIT

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
            cov = sig_init ** 2 * np.eye(2)
        else:
            cov = dead_reakon_covariance(distance, course['cog'], sig_d, sig_cog)

        yield {'longitude': lon,
               'latitude': lat,
               'covariance': cov}
        prev_lon = lon
        prev_lat = lat
        prev_time = time


def get_positions(mu, cov, ref):
    
    geod = pyproj.Geod(ellps="sphere")

    for i in range(ref.size / 2):
        sl = slice(2 * i, 2 * i + 2)
        lon, lat = ref[sl]
        one_cov = cov[sl, sl]
        angle = np.rad2deg(np.arctan2(*mu[sl]))
        dist = np.linalg.norm(mu[sl])
        new_lon, new_lat, _ = geod.fwd(lon, lat, angle, dist * 1000.)
        yield (new_lon, new_lat), one_cov


def plot_estimate(bm, mu, cov, ref, fig, ax, color='red'):
    geod = pyproj.Geod(ellps="sphere")

    for pos, one_cov in get_positions(mu, cov, ref):

        lon, lat = pos
        (w, h), eig_vecs = np.linalg.eig(one_cov)

        w = np.sqrt(w)
        h = np.sqrt(h)

        angle = np.rad2deg(np.arctan2(*np.dot(eig_vecs, [0., 1.])))

        lons, lats, _ = geod.fwd([lon, lon], [lat, lat],
                                 [angle, angle + 90.],
                                 [w * 1000, h * 1000])

        x_center, y_center = bm(lon, lat)
        x_axes, y_axes = map(np.array, bm(lons, lats))

        dists = np.sqrt(np.square(x_axes - x_center) + np.square(y_axes - y_center))
        ellipse = patches.Ellipse(bm(lon, lat), dists[0], dists[1], angle=-angle,
                                  edgecolor='k', facecolor=color,
                                  zorder=100, alpha=0.1)
        ax.add_patch(ellipse)
        plt.scatter(*bm(lon, lat), color=color)


def one_sight(time, lon, lat, body_name=None):
    obs = ephem.Observer()
    obs.lon = str(lon)
    obs.lat = str(lat)
    dt = pd.to_datetime(time)
    obs.date = dt.strftime('%Y/%m/%d %H:%M:%S')
    body_name = body_name or 'sun'
    body = _bodies[body_name]
    body.compute(obs)
    return np.rad2deg(body.alt), np.rad2deg(body.az), np.rad2deg(body.radius)


def build_observations(sights, times, ref):
    
    def one_sight_constraint(sight):
        i = np.nonzero(sight['time'] == times)[0]
        lon = np.asscalar(ref[2 * i][0])
        lat = np.asscalar(ref[2 * i + 1])

        alt, az, radius = one_sight(sight['time'], lon, lat, sight.get('body', None))
        obs_vect = np.zeros(ref.size)
        obs_vect[2 * i] = np.sin(np.deg2rad(az))
        obs_vect[2 * i + 1] = np.cos(np.deg2rad(az))

        obs_alt = reduction.corrected_altitude(sight['altitude'],
                                               height_m=0.,
                                               radius=radius)
        diff_nm = (obs_alt - alt) * 60.
        y = units.convert_scalar(diff_nm, 'nautical_mile', 'm') / 1000.
        return y, obs_vect
        
    
    obs = [one_sight_constraint(sight) for sight in sights]
    y, H = map(np.array, zip(*obs))
    R = SIG_ALT ** 2 * np.eye(y.size)
    return y, H, R


def plot_observations(y, H, R, ref, bm):

    geod = pyproj.Geod(ellps="sphere")
    for b, h, r in zip(y, H, np.diag(R)):

        lon, lat = ref[np.nonzero(h)]

        hx, hy = h[np.nonzero(h)]
        azim = np.rad2deg(np.arctan2(hx, hy))

        lop_lon, lop_lat, _ = geod.fwd(lon, lat, azim, b * 1000.)

        dists = np.linspace(-50000, 50000, 3)
        ret = np.array([geod.fwd(lop_lon, lop_lat, azim + 90, d)
                        for d in dists])

        lon_line = ret[:, 0]
        lat_line = ret[:, 1]
        bm.plot(lon_line, lat_line, color='red')


def plot_solution(sights, course, positions):

    lon_known = np.array([s.get('longitude', np.nan) for s in sights])
    lat_known = np.array([s.get('latitude', np.nan) for s in sights])

    lon_estimate, lat_estimate = map(np.array, zip(*[p for p, _ in positions]))
    lon_prior, lat_prior = ref.reshape((ref.size / 2, 2)).T
    lons = np.concatenate([lon_known, lon_estimate, lon_prior])
    lats = np.concatenate([lat_known, lat_estimate, lat_prior])
    lons = lons[np.isfinite(lons)]
    lats = lats[np.isfinite(lats)]

    fig, ax = plt.subplots(1, 1)
    bm = visualize.get_basemap(lons, lats, lat_pad=0.5, lon_pad=0.5, ax=ax)
    bm.plot(lon_known, lat_known, 'k')
    bm.scatter(lon_known, lat_known, color='k', s=25)
    plot_estimate(bm, mu, cov, ref, fig, ax, color='green')
    plot_estimate(bm, mu_post, cov_post, ref, fig, ax, color='blue')
    plot_observations(y, H, R, ref, bm)
    plt.show()


def toy_problem():

    lon, lat = [14., -25.]
    cog = 290.
    sog = 6.
    first_time = np.datetime64('2017-03-15 00:00:00')
    
    times = np.array([first_time + np.timedelta64(4 * i, 'h') for i in range(8)])

    courses = [{'time': first_time, 'cog': cog, 'sog': sog, 'longitude': lon, 'latitude': lat}]
    dr_course = list(dead_reakon(courses, times))

    def fake_sight(lon, lat, time):
        obs = ephem.Observer()
        obs.lon = str(lon)
        obs.lat = str(lat)
        dt = pd.to_datetime(time)
        obs.date = dt.strftime('%Y/%m/%d %H:%M:%S')
        sun = ephem.Sun()
        sun.compute(obs)
        alt = np.rad2deg(sun.alt)
        alt -= reduction.refraction(alt)
        alt -= 16.1 / 60.
        return {'time': time, 'altitude': alt,
                'longitude': lon, 'latitude': lat,
                }

    sights = [fake_sight(c['longitude'], c['latitude'], t)
              for c, t in zip(dr_course, times)]

    sights = [s for s in sights if s['altitude'] > 0.]

    return sights, courses



def estimate(sights, courses, include_now=False):

    times = itertools.chain([s['time'] for s in sights],
                            [co['time'] for co in courses])

    times = sorted(set(times))

    if include_now:
        times.append(np.datetime64(datetime.utcnow()))

    apriori_positions = list(dead_reakon(courses, times))

    realigned_sights = [[s for s in sights if s['time'] == t]
                        for t in times]
    realigned_sights = [s[0] if s else {} for s in realigned_sights]
    known_positions = np.array([[s.get('longitude', np.nan),
                                 s.get('latitude', np.nan)]
                                for s in realigned_sights])

    mu, cov, ref = build_state(apriori_positions)

    mses = np.linalg.norm(ref.reshape(known_positions.shape) - known_positions,
                          axis=1)

    mu_prior = mu.copy()
    ref_prior = ref.copy()
    cov_prior = cov.copy()

    for i in range(2):
        y, H, R = build_observations(sights, times, ref)

        ret = kalman_update(mu, cov, y, H, R)

        mu_post, cov_post, _, _ = ret

        new_ref = np.array([m for m, _ in get_positions(mu_post, cov_post, ref)]).reshape(-1)

        mu = np.zeros(mu.size)
        cov = cov_prior
        ref = new_ref

    lon_known = np.array([s.get('longitude', np.nan) for s in sights])
    lat_known = np.array([s.get('latitude', np.nan) for s in sights])

    pos_estimate = list(get_positions(mu_post, cov_post, ref))
    lon_estimate, lat_estimate = map(np.array, zip(*[p for p, _ in pos_estimate]))
    lon_prior, lat_prior = ref.reshape((ref.size / 2, 2)).T
    lons = np.concatenate([lon_known, lon_estimate, lon_prior])
    lats = np.concatenate([lat_known, lat_estimate, lat_prior])
    lons = lons[np.isfinite(lons)]
    lats = lats[np.isfinite(lats)]

    fig, ax = plt.subplots(1, 1)
    bm = visualize.get_basemap(lons, lats, lat_pad=0.5, lon_pad=0.5, ax=ax)
    bm.plot(lon_known, lat_known, 'k')
    bm.scatter(lon_known, lat_known, color='k', s=25)
#     plot_estimate(bm, mu, cov, ref, fig, ax, color='green')
    plot_estimate(bm, mu_post, cov_post, ref, fig, ax, color='blue')
    plot_observations(y, H, R, ref, bm)

    return times, get_positions(mu, cov_post, ref)



def main(args):
    sights = utils.read_sights(args.sights)
    courses = utils.read_courses(args.courses)

#     sights, courses = toy_problem()

    times, positions = list(estimate(sights, courses, include_now=args.now))

    print "----------------------------------------------------------"
    for t, (pos, cov) in zip(times, positions):
        lon_str = utils.decimal_to_degrees_minutes(pos[1])
        lat_str = utils.decimal_to_degrees_minutes(pos[0])
        max_err = units.convert_scalar(1000 * np.max(np.sqrt(np.linalg.eigvals(cov))),
                                       'm', 'nautical_mile')
        
        print "At %s you were here %s, %s +/- %.1f nm" % (t, lon_str, lat_str, max_err)
    print "----------------------------------------------------------"
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    tracker.py

    A tool to determine your latitude from a series of sights and DR.""")
    parser.add_argument('sights',
                        help=("path to a csv file holding sights."))
    parser.add_argument('courses',
                        help=("path to a csv of course changes."))
    parser.add_argument('--now', default=False, action="store_true")
    args = parser.parse_args()
    main(args)
