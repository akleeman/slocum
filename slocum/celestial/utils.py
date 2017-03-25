import csv
import numpy as np
import pandas as pd


def degrees_minutes_to_decimal(deg_min_str):
    degrees, minutes = [np.float(x.strip("'"))
                        for x in deg_min_str.split('_', 1)]
    if degrees == 0.:
        return minutes / 60.
    else:
        return np.sign(degrees) * (np.abs(degrees) + minutes / 60.)


def decimal_to_degrees_minutes(angle):
    degrees = np.sign(angle) * np.floor(np.abs(angle))
    minutes = np.abs(angle - degrees) * 60
    return "%d_%.2f'" % (degrees, minutes)


def maybe_convert_to_decimal(deg_min_str):
    if not '_' in deg_min_str:
        return np.float(deg_min_str)
    else:
        return degrees_minutes_to_decimal(deg_min_str)


def parse_angle(angle_string, time_utc=None):
    """
    Takes an angle and converts it into decimal form, two
    angles can be given for starts/ends of hours in which
    case the time is used to interpolate between them.
    """
    if not isinstance(angle_string, basestring) and np.isnan(angle_string):
        return np.nan
    if ':' in angle_string:
        if time_utc is None:
            raise ValueError("In order to interpolate between angles"
                             " a time needs to be given.")
        start, end = map(maybe_convert_to_decimal,
                         angle_string.split(':'))
        time_stamp = pd.to_datetime(time_utc)
        fraction = time_stamp.minute / 60. + time_stamp.second / 3600.

        if np.abs(start - end) > 180.:
            if start > end:
                end = end + 360.
            else:
                start = start + 360.

        return np.mod(start + fraction * (end - start), 360.)
    else:
        return maybe_convert_to_decimal(angle_string)


def parse_one_sight(sight):
    """
    Pull out all the information from one line in a csv of sights.
    Not all the information may be present, in which case nans
    are filled in.
    """
    time = np.datetime64(sight.get('utc', np.nan))
    gha = parse_angle(sight.get('gha', np.nan), time)
    dec = parse_angle(sight.get('dec', np.nan), time)
    lat = maybe_convert_to_decimal(sight.get('latitude', np.nan))
    lon = maybe_convert_to_decimal(sight.get('longitude', np.nan))
    alt = maybe_convert_to_decimal(sight.get('altitude', np.nan))
    radius = maybe_convert_to_decimal(sight.get('radius', np.nan))
    body = sight.get('body', 'sun')
    return {'time': time,
            'gha': gha,
            'declination': dec,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'radius': radius}


def parse_one_course(course):
    """
    Pull out all the information from one line in a csv of courses.
    Not all the information may be present, in which case nans
    are filled in.
    """
    time = np.datetime64(course.get('utc', np.nan))
    cog = maybe_convert_to_decimal(course.get('cog', np.nan))
    sog = maybe_convert_to_decimal(course.get('sog', np.nan))
    lat = maybe_convert_to_decimal(course.get('latitude', np.nan))
    lon = maybe_convert_to_decimal(course.get('longitude', np.nan))
    return {'time': time,
            'cog': cog,
            'latitude': lat,
            'longitude': lon,
            'sog': sog, }


def read_csv(file_name, expected_fields = []):
    """
    Reads in a csv of sights into a list of dictionaries.
    """

    with open(file_name, 'r') as f:
        reader = csv.DictReader(f)
        expected_names = set(expected_fields)
        if len(expected_names.difference(reader.fieldnames)):
            raise ValueError("Expected fields with names %s, got %s, missing %s"
                             % (expected_names, reader.fieldnames,
                                expected_names.difference(reader.fieldnames)))
        data = list(reader)

    return data



def read_sights(file_name):
    data = read_csv(file_name, ['utc', 'altitude'])
    return [parse_one_sight(sight) for sight in data]
    

def read_courses(file_name):
    data = read_csv(file_name, ['cog', 'sog'])
    return [parse_one_course(course) for course in data]
