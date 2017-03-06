import pyproj
import numpy as np

from slocum.lib import units


def altitude(declination, latitude, lha):
    """
    Returns Hc the altitude of a body given the declination, latitude
    and local hour angle.
    """
    declination = np.deg2rad(declination)
    latitude = np.deg2rad(latitude)
    lha = np.deg2rad(lha)

    alt = np.arcsin(np.sin(declination) * np.sin(latitude) +
                    np.cos(latitude) * np.cos(declination) * np.cos(lha))
    return np.rad2deg(alt)


def zenith(declination, latitude, altitude, lha):
    """
    Computes Z, the zenith of a body.
    """
    declination = np.deg2rad(declination)
    latitude = np.deg2rad(latitude)
    altitude = np.deg2rad(altitude)
    lha = np.deg2rad(lha)

    num = (np.sin(declination) - np.sin(latitude) * np.sin(altitude))
    z = np.arccos(num / (np.cos(latitude) * np.cos(altitude)))
    z = np.rad2deg(z)
    if z < 0.:
        z += 180.
    if lha < 0.:
        z += 180.
    return np.mod(z, 360.)


def local_hour_angle(gha, longitude):
    """
    Local Hour Angle
    """
    return np.mod(gha + longitude + 180., 360.) - 180.


def refraction(altitude):
    """
    Correction due to refraction.
    """
    altitude = np.deg2rad(altitude)
    return 0.96 / 60. * np.tan(altitude)


def dip(height_m):
    """
    Correction due to fact that the observer is not on the
    earth's surface.
    """
    return 1.76 / 10. * np.sqrt(height_m)


def corrected_altitude(altitude, height_m, radius):
    """
    Apply corrections such as dip and refraction to an altitude.
    """
    return altitude + dip(height_m) + refraction(altitude) + radius


def line_of_position(utc, declination, gha, longitude, latitude, alt,
                     height_m=0.0, radius=16.1 / 60.):

    lha = local_hour_angle(gha, longitude)
    expected_alt = altitude(declination, latitude, lha)
    z = zenith(declination, latitude, expected_alt, lha)
    corrected_alt = corrected_altitude(alt, height_m, radius)

    geod = pyproj.Geod(ellps="sphere")
    alt_diff = (expected_alt - corrected_alt)
    dist_m = units.convert_scalar(alt_diff * 60, 'nautical_mile', 'm')
    lon, lat, _ = geod.fwd(longitude, latitude, z, dist_m)
    return lon, lat, z
