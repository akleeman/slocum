import pyproj
import numpy as np


def angle_normalize(a, degrees=True):
    if degrees:
        a = np.mod(a + 180, 360) - 180
    else:
        a = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return a


def angle_diff(a, b=None, degrees=True):
    """
    Computes the angular difference between angle a
    and angle b.  Equivalent to a - b, but all results
    are modulo 180 degrees.
    """
    # if b is one this performs like np.diff but for angles
    if b is None:
        return angle_diff(a[1:], a[:-1], degrees)
    # decided how to wrap based on the degrees flag.
    return angle_normalize(a - b, degrees=degrees)


def angle_add(a, b, degrees=True):
    return angle_normalize(a + b, degrees=degrees)


def angle_sort(x, degrees=True):
    """
    Sorts using angle_diff so that the resulting list of angles will be ascending
    even if they cross 0, 180, 360 etc ...

    Example:

        > lons = np.mod(np.linspace(170, 190, 11) + 180, 360) - 180
        > print lons
            array([ 170.,  172.,  174.,  176.,  178., -180., -178., -176., -174.,
                   -172., -170.])
        > print angle_sort(lons)
            array([ 170.,  172.,  174.,  176.,  178., -180., -178., -176., -174.,
                   -172., -170.])

    """
    angle_cmp = lambda x, y: int(np.sign(angle_diff(x, y, degrees=degrees)))
    return np.array(sorted(x, cmp=angle_cmp))


def geographic_distance(lon0, lat0, lon1, lat1, ellps="sphere"):
    """
    Computes the distance (in meters) between two points
    assuming a particular earth ellipse model.
    """
    geod = pyproj.Geod(ellps=ellps)
    return geod.inv(lon0, lat0, lon1, lat1)[-1]


def vector_to_radial(u, v, orientation='to'):
    """
    Converts from a vector variable with zonal (u) and
    meridianal (v) components to magnitude and direction
    from north.
    """
    assert orientation in ['from', 'to']
    # convert to magnitudes
    magnitude = np.sqrt(np.power(u, 2) + np.power(v, 2))
    direction = np.arctan2(u, v)
    if orientation == 'from':
        # note this is like: direction + pi but with modulo between
        direction = np.mod(direction + 2 * np.pi, 2 * np.pi) - np.pi
    return magnitude, direction


def radial_to_vector(magnitude, direction, orientation='to'):
    """
    Converts from a radial variable defined by a magnitude and
    direction from north to zonal (u) and meridianal (v) components.
    """
    assert orientation in ['from', 'to']
    v = np.cos(direction) * magnitude
    u = np.sin(direction) * magnitude
    if orientation == "from":
        v = -v
        u = -u
    return u, v
