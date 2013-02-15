"""
The S.V. Spray was a 36-foot-9-inch (11.20 m) oyster sloop rebuilt by Joshua
Slocum and used by him to sail single-handed around the world, the first voyage
of its kind. The Spray was lost with Captain Slocum aboard in 1909, while
sailing from Vineyard Haven, Massachusetts, on the island of Martha's Vineyard,
to South America.

spray.py holds tools related to sailboat dynamics
"""
import copy
import numpy as np
import coards
import logging
import datetime
import itertools

import matplotlib.pyplot as plt

import sl.objects.conventions as conv

from sl.lib import datelib, plotlib, numpylib
from sl.objects import objects, core

logging.basicConfig(level=logging.DEBUG)

_max_pointing = 0.3490658503988659 # 30/180*pi
_max_boat_speed = 6
_min_wind_speed = 4
_req_wind_speed = 10
_earth_radius = 3440.07 # in nautical miles
_dangerous_wind_speed = 30

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
    if wind.speed > _min_wind_speed and wind.speed <= _req_wind_speed:
        speed = np.sqrt((wind.speed - _min_wind_speed) /
                        (_req_wind_speed - _min_wind_speed))
        return directional_max_speed(deg_off_wind) * speed
    elif wind.speed > _req_wind_speed:
        return directional_max_speed(deg_off_wind)
    elif wind.speed > _dangerous_wind_speed:
        if deg_off_wind <= 60:
            return 0.0
        elif deg_off_wind <= 100:
            return 0.25*_max_boat_speed
        else:
            return _max_boat_speed
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

def trajectory(legs):
    """
    Turns a list of route legs into a trajectory
    """
    legs = list(legs)
    nlegs = len(legs)
    traj = core.Data()
    traj.create_coordinate(conv.STEP, np.arange(nlegs))

    variables = legs[0].variables.keys()
    for var in variables:
        traj.create_variable(var, dim = (conv.STEP,),
                             data=np.zeros((nlegs,)),
                             attributes=legs[0][var].attributes)

    for (it, step), leg in zip(traj.iterator(conv.STEP, views=True), legs):
        for var in variables:
            step[var].data[:] = np.asscalar(leg[var].data)
    return traj

class OutOfDataException(Exception):
    pass

def passage(waypoints, start_date, weather, fast=False, accuracy=0.5):
    """
    Simulates a passage following waypoints having started on a given date.

    waypoints - a list of at least two waypoints, the first of which is the
        start location.
    start_date - the date the passage is started
    """
    waypoints = copy.deepcopy(waypoints)
    def has_wind(obj):
        return conv.UWND in obj.variables and conv.VWND in obj.variables
    wind_data = [x for x in weather if has_wind(x)]
    #assert len(wind_data) == 1
    wind_data = wind_data[0]

    data_units = wind_data[conv.TIME].attributes['units']

    dates = datelib.from_udvar(wind_data[conv.TIME])
    data_start = dates[0]
    data_end = dates[-1]
    print "Forecast spans %s to %s" % (data_start, data_end)
    if start_date < data_start or start_date >= data_end:
        msg = "start date %s is outside the range of the data [%s, %s]"
        raise ValueError(msg % (start_date, data_start, data_end))

    def iter_valid():
        prev_date = None
        for t, w in wind_data.iterator(dim=conv.TIME):
            if conv.ENSEMBLE in w.variables:
                w = w.squeeze(conv.ENSEMBLE)
            w = w.squeeze(conv.TIME)
            date = datelib.from_udvar(t)[0]
            if not prev_date:
                prev_date = date
                yield date, w
                continue
            mid_date = prev_date + datetime.timedelta(seconds=0.5*datelib.seconds(date - prev_date))
            yield mid_date, w
            prev_date = date
        last_date = date + datetime.timedelta(seconds=datelib.seconds(date - mid_date))
        yield last_date, w

    def iter_times():
        valid_dates = iter_valid()
        prev_date, prev_wx = valid_dates.next()
        for date, wx in valid_dates:
            if date > start_date:
                break
            prev_date, prev_wx = date, wx
        yield prev_date, prev_wx
        yield date, wx
        for date, wx in valid_dates:
            yield date, wx

    def legs(waypoints):
        waypoints = list(waypoints)
        here = waypoints.pop(0)

        all_times = list(iter_times())
        time_iter = iter(all_times)
        fcst_time, wx = time_iter.next()
        now = start_date
        soon, next_wx = time_iter.next()
        dt = soon - now

        nan_obj = core.Data()
        units, init_time = datelib.to_udvar([now], units=data_units)
        nan_obj.create_coordinate(conv.TIME, init_time,
                               attributes={'units':units})

        variables = [conv.UWND, conv.VWND, conv.LAT, conv.LON,
                     conv.BEARING, conv.HEADING, conv.SPEED, conv.DISTANCE,
                     conv.RELATIVE_WIND, conv.HOURS, conv.MOTOR_ON,
                     conv.WIND_SPEED]
        [nan_obj.create_variable(x, dim=(conv.TIME,), data=np.array([np.nan]))
         for x in variables]

        def get_next():
            """ a wrapper around time_iter which throws an out of data exception"""
            try:
                return time_iter.next()
            except StopIteration:
                raise OutOfDataException("Ran out of data, final time: %s" % str(now))

        for destination in waypoints:
            try:
                while not here == destination:
                    # interpolate the weather at the current lat lon
                    local_wx = wx.interpolate(lat=here.lat, lon=here.lon, fast=fast)
                    # fill with local weather
                    local_vars = local_wx.variables.iteritems()
                    obj = copy.deepcopy(nan_obj)
                    [obj[k].data.put(0, np.asscalar(v.data)) for k, v in local_vars if k in obj.variables]
                    obj[conv.TIME].data[0] = datelib.to_udvar([now], units)[1]
                    obj[conv.HOURS].data[0] = datelib.hours(dt)
                    # determine the bearing (following a rhumbline) between here and the end
                    bearing = rhumbline_bearing(here, destination)
                    # TODO: actually compute the heading
                    heading = bearing
                    # get the wind and use that to compute the boat speed
                    wind = objects.Wind(np.asscalar(obj[conv.UWND].data),
                                        np.asscalar(obj[conv.VWND].data))
                    speed = boat_speed(wind, bearing)
                    obj[conv.MOTOR_ON].data[0] = (wind.speed <= 5.)
                    speed = 5.5 if obj[conv.MOTOR_ON].data[0] else speed
                    distance = speed * datelib.hours(dt)
                    # given our speed how far can we go in one timestep?
                    remaining = rhumbline_distance(here, destination)
                    time_string = ("time: %s " % now)
                    if distance > max(0., remaining - accuracy):
                        here = destination
                        required_time = int(datelib.hours(dt) * datelib._seconds_in_hour * remaining / distance)
                        obj[conv.HOURS].data[0] *= (remaining / distance)
                        dt = datetime.timedelta(seconds=required_time)
                        now = now + dt
                        distance = remaining
                        while now > soon:
                            # no need to assign now since we know its past soon
                            _, wx = soon, next_wx
                            soon, next_wx = get_next()
                    else:
                        # and once we know how far, where does that put us in terms of lat long
                        here = rhumbline_path(here, bearing)(distance)
                        dt = soon - now
                        now, wx = soon, next_wx
                        soon, next_wx = get_next()
                    obj[conv.LAT].data[0] = here.lat
                    obj[conv.LON].data[0] = here.lon
                    obj[conv.DISTANCE].data[0] = distance
                    obj[conv.BEARING].data[0] = bearing
                    obj[conv.HEADING].data[0] = heading
                    obj[conv.SPEED].data[0] = speed
                    rel_wind = np.abs(heading - wind.dir)
                    if rel_wind > np.pi:
                        rel_wind = 2.*np.pi - rel_wind
                    obj[conv.RELATIVE_WIND].data[0] = rel_wind
                    obj[conv.WIND_SPEED].data[0] = wind.speed
                    #print ('%s wind: %4s (%4.1f) @ %6.1f knots \t %6.1f miles in %4.1f hours @ %6.1f knots'
                    #             % (time_string, wind.readable, wind.dir, wind.speed, distance, datelib.hours(dt), speed))
                    yield obj
                    if remaining < accuracy:
                        break
            except OutOfDataException, e:
                pos_only = copy.deepcopy(nan_obj)
                pos_only[conv.LAT].data[0] = destination.lat
                pos_only[conv.LON].data[0] = destination.lon
                yield pos_only

    def iter_waypoints():
        wps = copy.deepcopy(waypoints)
        here = wps.pop(0)
        next = wps.pop(0)
        distance = rhumbline_distance(here, next)
        yield here
        while distance > 60:
            bearing = rhumbline_bearing(here, next)
            here = rhumbline_path(here, bearing)(60)
            distance = rhumbline_distance(here, next)
            yield here

    waypoints = list(iter_waypoints())
    #print '\n'.join(['%f, %f' % (x.lat, x.lon) for x in waypoints])
    traj = trajectory(legs(waypoints))
    return traj

def ensemble_passage(waypoints, start_date, weather, *args, **kwdargs):
    passages = [passage(waypoints, start_date, [wx], *args, **kwdargs) for wx in weather]
    return normalize_ensemble_passages(passages)

def iterroutes(start, end, start_date, ensemble_weather, resol=None):
    if resol is None:
        resol = 0.25*min(np.abs(start.lat - end.lat), np.abs(start.lon - end.lon))
        print resol
    else:
        resol = float(resol)

    def issafe(route):
        "returns a boolean indicating the route was a safe one"
        return np.max([x['max_wind_speed'] for x in route]) <= _dangerous_wind_speed

    def simulate_passage(x, start_date):
        return ensemble_passage(x, start_date, weather=ensemble_weather, fast=False)

    def chain_passages(x, y):
        if x is None and y is None:
            raise ValueError("expected one non None argument")
        if x is None:
            return y
        if y is None:
            return x
        if x.dimensions[conv.ENSEMBLE] != y.dimensions[conv.ENSEMBLE]:
            raise ValueError("expected x and y to have same number ensembles")
        def iterensembles(x, y):
            if x.dimensions[conv.ENSEMBLE] == 1:
                x = x.view(slice(0, x[conv.NUM_STEPS].data[0]), conv.STEP)
                y = y.view(slice(0, y[conv.NUM_STEPS].data[0]), conv.STEP)
                nsteps = x[conv.NUM_STEPS].data + y[conv.NUM_STEPS].data
                x[conv.NUM_STEPS].data[:] = nsteps
                y[conv.NUM_STEPS].data[:] = nsteps
                yield objects.merge([x, y], conv.STEP)
            else:
                for (_, w), (_,z) in zip(x.iterator(conv.ENSEMBLE), y.iterator(conv.ENSEMBLE)):
                    yield list(iterensembles(w, z))[0]

        return normalize_ensemble_passages(iterensembles(x, y))

    def route_filter(x):
        "Returns the passage summaries for a route through x"
        summaries = [summarize_passage(p) for p in passages]
        return issafe(summaries)

    def route_morph(waypoints, prev_passage = None, resol = 1.):
        start = waypoints[-1]
        if prev_passage is None:
            new_start_date = start_date
            prev_passage = None
        else:
            prev_times = [time[steps - 1] for time, steps in
                            zip(prev_passage[conv.TIME].data.T,
                                prev_passage[conv.NUM_STEPS].data)]
            time_var = core.Variable(dim=(conv.TIME,),
                                     data=[np.mean(prev_times)],
                                     attributes = prev_passage[conv.TIME].attributes)
            new_start_date = datelib.from_udvar(time_var)[0]

        # create a set of waypoints with a detour along the latitude line
        lat_dir = np.sign(start.lat - end.lat)
        lat_delta = min(resol, np.abs(start.lat - end.lat)) * lat_dir
        lat_waypoint = objects.LatLon(start.lat - lat_delta, start.lon)
        # create a set of waypoints with a detour along the longitude line
        lon_dir = np.sign(start.lon - end.lon)
        lon_delta = min(resol, np.abs(start.lon - end.lon)) * lon_dir
        lon_waypoint = objects.LatLon(start.lat, start.lon - lon_delta)
        diag_waypoint = objects.LatLon(start.lat - lat_delta, start.lon - lon_delta)

        if np.abs(start.lat - end.lat) < resol and np.abs(start.lon - end.lon) < resol:
            print 'end'
            if start.lat - end.lat < 1e-4 and start.lon - end.lon < 1e-4:
                yield waypoints, prev_passage
            else:
                rhumbline_passage = simulate_passage([start, end], new_start_date)
                rhumbline_waypoints = list(itertools.chain(waypoints, [end]))
                yield rhumbline_waypoints, chain_passages(prev_passage, rhumbline_passage)
        else:
            print '.'
            # yield passages along the latitude
            if lat_delta > resol/10.:
                lat_passage = simulate_passage([start, lat_waypoint], new_start_date)
                lat_waypoints = list(itertools.chain(waypoints, [lat_waypoint]))
                for wpts, psgs in route_morph(lat_waypoints,
                                              chain_passages(prev_passage, lat_passage),
                                              resol=resol):
                    yield wpts, psgs
            if lon_delta > resol/10.:
                # yield passages along the longitude
                lon_passage = simulate_passage([start, lon_waypoint], new_start_date)
                lon_waypoints = list(itertools.chain(waypoints, [lon_waypoint]))
                for wpts, psgs in route_morph(lon_waypoints,
                                              chain_passages(prev_passage, lon_passage),
                                              resol=resol):
                    yield wpts, psgs
            if lat_delta > resol/10. and lon_delta > resol/10.:
                # yield passages along the diagonal
                diag_passage = simulate_passage([start, diag_waypoint], new_start_date)
                diag_waypoints = list(itertools.chain(waypoints, [diag_waypoint]))
                for wpts, psgs in route_morph(diag_waypoints,
                                              chain_passages(prev_passage, diag_passage),
                                              resol=resol):
                    yield wpts, psgs

    return route_morph([start], resol=resol)

def optimal_passage(iter_routes):

    def idealness(p):
        "returns a scalar factor representing the idealness of a route, smaller is better"
        summaries = [summarize_passage(ens) for x, ens in p.iterator(conv.ENSEMBLE)]
        time = np.mean([x['hours'] for x in summaries])
        motor_hours = np.mean([x['motor_hours'] for x in summaries])
        upwind_hours = np.mean([x['upwind_hours'] for x in summaries])
        return motor_hours + upwind_hours
        #dist = np.mean([x['distance'] for x in route])
        #pct_upwind = np.mean([x['pct_upwind'] for x in route])
        #return (time - avg_times)/avg_times + (avg_distances/dist - 1.) + pct_upwind

    ranked_routes = sorted([(idealness(r), (w, r)) for w, r in iter_routes],
                           key=lambda x : x[0], reverse=True)
    waypoints = [x[1][0] for x in ranked_routes]
    routes = [x[1][1] for x in ranked_routes]
    idealness = [x[0] for x in ranked_routes]
    colors = [plt.cm.RdYlGn(1. - (x - min(idealness))/(max(idealness) - min(idealness))) for x in idealness]
    print "optimal waypoints:"
    print ":".join("%s,%s" % (x.lat, x.lon) for x in waypoints[np.argmin(idealness)])
    plotlib.plot_routes(routes, colors)
    return routes[np.argmin(idealness)]

def summarize_passage(passage):
    ret = {}
    if conv.ENSEMBLE in passage.dimensions and passage.dimensions[conv.ENSEMBLE] != 1:
        raise ValueError("expected a single ensemble dim")
    passage = passage.view(slice(0, passage[conv.NUM_STEPS].data[0]), conv.STEP)
    times = datelib.from_udvar(passage[conv.TIME])
    start_time = times[0][0]
    end_time = times[-1][0]
    ret['hours'] = datelib.hours(end_time - start_time)
    wind = passage[conv.WIND_SPEED].data
    ret['max_wind_speed'] = max(wind)[0]
    ret['min_wind_speed'] = min(wind)[0]
    ret['avg_wind_speed'] = np.mean(wind)
    ret['motor_hours'] = sum(passage['motor_on'].data * passage['hours'].data)[0]
    upwind_sail = np.logical_and(np.logical_not(passage['motor_on']),
                                 passage['rel_wind'] < (np.pi / 2))
    ret['upwind_hours'] = sum(passage['hours'].data * upwind_sail)[0]
    return ret
