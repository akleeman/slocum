import numpy as np

from lib import objects

earth_radius = 3440.07 # in nautical miles

def apparent_wind(course, true_wind):
    """
    Given a course (speed and bearing) and a true wind this returns the
    apparent wind as seen from the boat
    """
    boat_u = course.speed * np.sin(course.bearing)
    boat_v = course.speed * np.cos(course.bearing)

    # the wind relative to the due north
    relative_wind = objects.Wind(true_wind.u - boat_u, true_wind.v - boat_v)
    # the direction is simply the difference in angles
    dir = relative_wind.dir - course.bearing
    # its easiest to decompose back into u and v apparent directions
    return objects.Wind(u=-relative_wind.speed * np.sin(dir),
                        v=-relative_wind.speed * np.cos(dir))

def rhumbline_bearing(a, b):
    """
    Gives you the bearing of a rhumbline between two points a and b
    """
    # first we convert to mercator coordinates
    a = a.as_rad()
    b = b.as_rad()
    delta_lon = b.lon - a.lon
    stretched_lat = np.log(np.tan(b.lat/2.+np.pi/4.)/np.tan(a.lat/2.+np.pi/4.))

    return np.arctan2(delta_lon, stretched_lat)

def rhumbline_distance(a, b):
    """
    Gives the distance between two points a and b following a rhumbline
    """
    a = a.as_rad()
    b = b.as_rad()
    stretched_lat = np.log(np.tan(b.lat/2.+np.pi/4.)/np.tan(a.lat/2.+np.pi/4.))
    delta_lat = (b.lat - a.lat)
    delta_lon = (b.lon - a.lon)
    if delta_lat == 0.0:
        q = np.cos(b.lat)
    else:
        q = delta_lat / stretched_lat

    return np.sqrt(np.power(delta_lat, 2.) + np.power(q*delta_lon, 2.)) * earth_radius

def rhumbline_path(a, bearing):
    # first we convert to mercator coordinates
    a = a.as_rad()
    phia = np.log(np.tan(a.lat/2.+np.pi/4.))

    def f(distance):
        if distance == 0.0:
            return a.as_deg()
        alpha = distance/earth_radius
        delta_lat = alpha * np.cos(bearing)
        latb = a.lat + delta_lat
        stretched_lat = np.log(np.tan(latb/2.+np.pi/4.)) - phia
        if delta_lat:
            q = np.cos(latb)
        else:
            q = delta_lat / stretched_lat
        delta_lon = alpha * np.sin(bearing) / q
        return objects.LatLon(np.rad2deg(latb), np.rad2deg(a.lon + delta_lon))

    return f