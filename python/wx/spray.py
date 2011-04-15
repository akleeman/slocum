"""
The S.V. Spray was a 36-foot-9-inch (11.20 m) oyster sloop rebuilt by Joshua
Slocum and used by him to sail single-handed around the world, the first voyage
of its kind. The Spray was lost with Captain Slocum aboard in 1909, while
sailing from Vineyard Haven, Massachusetts, on the island of Martha's Vineyard,
to South America.

spray.py holds tools related to sailboat dynamics
"""
import numpy as np
import coards

import wx.lib.conventions as conv
from wx.objects import objects

_max_pointing = 0.3490658503988659 # 30/180*pi
_max_boat_speed = 6
_min_boat_speed = 3
_req_boat_speed = 15
_max_boat_speed = 35
_earth_radius = 3440.07 # in nautical miles

def hours(timedelta):
    return float(timedelta.days * 24 + timedelta.seconds / SECONDS_IN_HOUR)

def directional_max_speed(deg_off_wind):
    """
    Gives the max boat speed given the degrees off the wind
    """
    if deg_off_wind > _max_pointing:
        return _max_boat_speed
    else:
        return (0.5 + 0.5* deg_off_wind / _max_pointing) * _max_boat_speed

def boat_speed(wind, bearing):
    """
    Returns a scalar indicating the boats speed in knots given an apparent
    wind and a bearing.
    """
    deg_off_wind = np.abs(wind.dir - bearing)
    if deg_off_wind > np.pi: deg_off_wind = 2*np.pi - deg_off_wind
    if wind.speed > _min_boat_speed and wind.speed <= _req_boat_speed:
        speed = np.sqrt((wind.speed - _min_boat_speed) /
                        (_req_boat_speed - _min_boat_speed))
        return directional_max_speed(deg_off_wind) * speed
    elif wind.speed > _min_boat_speed and wind.speed <= _max_boat_speed:
        return directional_max_speed(deg_off_wind)
    else:
        return 0

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

    return _earth_radius * np.sqrt(np.power(delta_lat, 2.) +
                                   np.power(q*delta_lon, 2.))

def rhumbline_path(a, bearing):
    """
    returns a function that takes a distances and returns your lat long after
    traveling that distance along the rhumbline given by 'bearing'
    """
    # first we convert to mercator coordinates
    a = a.as_rad()
    phia = np.log(np.tan(a.lat/2.+np.pi/4.))

    def f(distance):
        if distance == 0.0:
            return a.as_deg()
        alpha = distance/_earth_radius
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

def passage(waypoints, start_date, weather):
    """
    Simulates a passage following waypoints having started on a given date.

    waypoints - a list of at least two waypoints, the first of which is the
        start location.
    start_date - the date the passage is started
    wxfunc - a function with signature f(time, lat, lon) that must return a
        pupynere-like object containing at least uwnd, vwnd
    """
    waypoints = list(waypoints)
    here = waypoints.pop(0)

    base_data = [x for x in weather if conv.UWND in x.variables and conv.VWND in x.variables]
    assert len(base_data) == 1
    base_data = base_data[0]

    units = base_data[conv.TIME].attributes['units']
    time_iter = base_data.iterator(dim=conv.TIME)
    time_iter = [coards.from_udunits(t.data[0], units) for t, y in time_iter]

    now = next_time()
    soon = next_time()
    dt = soon - now

    def weather(time, lat, lon):
        return base_data.interpolate(time=time, lat=lat, lon=lon)

    weather(now, base_data['lat'].data[1], base_data['lon'].data[1])
    import pdb; pdb.set_trace()
    try:
        for destination in waypoints:
            while not here == destination:
                # interpolate the weather in wx_fields at the current lat lon

                local_wx = wx.interpolate(lat=here.lat, lon=here.lon)
                uwnd = np.asscalar(local_wx['uwnd'].data)
                vwnd = np.asscalar(local_wx['vwnd'].data)
                # determine the bearing (following a rhumbline) between here and the end
                bearing = rhumbline_bearing(here, destination)
                # get the wind and use that to compute the boat speed
                wind = objects.Wind(uwnd, vwnd)
                speed = max(boat_speed(wind, bearing), 1.0)
                course = objects.Course(here, speed, bearing, bearing)
                rel_wind = np.abs(course.heading - wind.dir)
                if rel_wind > np.pi:
                    rel_wind = 2.*np.pi - rel_wind
                # given our speed how far can we go in one timestep?
                distance = speed * hours(dt)
                remaining = rhumbline_distance(here, destination)
                if distance > remaining:
                    here = destination
                    required_time = int(hours(dt) * SECONDS_IN_HOUR * remaining / distance)
                    now = now + datetime.timedelta(seconds=required_time)
                    dt = soon - now
                    distance = remaining
                else:
                    # and once we know how far, where does that put us in terms of lat long
                    here = rhumbline_path(here, bearing)(distance)
                    now, wx = soon, next_wx
                    soon, next_wx = next_time()
                dt = soon - now
                logging.debug('wind: %4s (%4.1f) @ %6.1f knots \t %6.1f miles in %4.1f hours @ %6.1f knots'
                             % (wind.readable, wind.dir, wind.speed, distance, hours(dt), speed))

                yield objects.Leg(course, now, wind, distance, rel_wind, wx)
    except StopIteration:
        logging.error("Ran out of data!")


def optimal_passage(start, end, start_date, forecasts, resol=5):
    # get the corners
    c1 = objects.LatLon(start.lat, end.lon)
    c2 = objects.LatLon(end.lat, start.lon)
    forecasts = list(forecasts)

    def issafe(route):
        "returns a boolean indicating the route was a safe one"
        return np.max([x['max_wind'] for x in route]) <= MAX_WIND_SPEED

    def route(x):
        "Returns the passage summaries for a route through x"
        passages = simulate_passages([start, x, end], start_date, forecasts)
        summaries = [summarize_passage(passage) for passage in passages]
        if issafe(summaries):
            return summaries
        else:
            return None
        return summaries

    waypoints = [objects.LatLon(x*c1.lat + (1.-x)*c2.lat, x*c1.lon + (1.-x)*c2.lon) for x in np.arange(0., 1., step=1./resol)]
    routes = [(x, route(x)) for x in waypoints]
    avg_times = np.mean([np.mean([x['hours'] for x in route]) for x, route in routes])
    avg_distances = np.mean([np.mean([x['distance'] for x in route]) for x, route in routes])
    def idealness(route):
        "returns a scalar factor representing the idealness of a route, smaller is better"
        time = np.mean([x['hours'] for x in route])
        dist = np.mean([x['distance'] for x in route])
        pct_upwind = np.mean([x['pct_upwind'] for x in route])
        return (time - avg_times)/avg_times + (avg_distances/dist - 1.) + pct_upwind
    idealness = [idealness(route) for x, route in routes]

    return routes[np.argmin(idealness)][0]

def summarize_passage(passage):
    ret = {}
    passage = list(passage)

    ret['hours'] = hours(passage[-1].time - passage[0].time)
    ret['distance'] = rhumbline_distance(passage[-1].course.loc,
                                                    passage[0].course.loc)
    wind = [x.wind.speed for x in passage]
    ret.update({'min_wind':np.min(wind), 'max_wind':np.max(wind), 'avg_wind':np.mean(wind)})
    dist = [x.distance for x in passage]
    ret.update({'min_dist':np.min(dist), 'max_dist':np.max(wind), 'avg_dist':np.mean(wind)})
    ret['pct_upwind'] = [x.rel_wind_dir < np.pi/4 for x in passage]
    return ret
