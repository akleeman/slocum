import os
import coards
import numpy as np
import logging
import datetime

from dateutil.relativedelta import relativedelta

import sl.objects.conventions as conv

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

_seconds_in_hour = 3600.

def remove_old_files(files, days_old=3):
    """
    Removes any file in list if the file was created more than 'days_old' days ago
    """
    files = list(files)
    remove_if_before = datetime.datetime.now() - datetime.timedelta(days=days_old)
    created = map(datetime.datetime.fromtimestamp, map(os.path.getctime, files))
    for f, t in zip(files, created):
        if t < remove_if_before:
            logger.debug('removing %d day old file %s' %
                          ((datetime.datetime.now() - t).days, f))
            os.remove(f)

def seconds(timedelta):
    return float(timedelta.days * 24 * _seconds_in_hour + timedelta.seconds)

def hours(timedelta):
    return seconds(timedelta) / _seconds_in_hour

def from_coards(times, units):
    out = np.empty(times.shape, dtype=object)
    # unfortunately coards.py does not interpret months in the same way
    # as either nostra, cdo or ncdump.  it uses a number of seconds rather
    # than a relative timedelta.  the relativedelta from dateutil will do
    # the right thing for days and months -- it's also faster because the
    # value of origin is only parsed out once rather than N times in the
    # the loop
    unit, origin = units.lower().split(' since ')
    origin = coards.parse_date(origin)

    for (i, x) in np.ndenumerate(times):
        if np.isnan(x):
            out[i] = None
        else:
            out[i] = origin + relativedelta(**{unit: int(x)})

    return out

def from_udvar(timevar):
    """
    Takes a time variable which is assumed to follow coards conventions
    and returns a numpy vector containing all the datetimes represented
    by the timevar.
    """
    if conv.UNITS not in timevar.attributes:
        raise ValueError("Time variables must include units")
    return from_coards(timevar.data, timevar.attributes[conv.UNITS])

def to_udvar(dates, units=None):
    """
    Converts a list of dates into a tuple of (units, data) that conform to
    the coards conventions.

    Parameters
    ----------
    dates : list_like, array_like, or datetime.datetime
        A datetime.datetime instance or a list/array of instances. An
        exception is raised if any element is not a datetime.datetime
        instance.
    units : string, optional
        A udunits-style units string. If None (default), the string
        "days since yyyy-mm-dd" is used, where the date corresponds to
        the minimum of dates.

    Returns
    -------
    units : string
        A udunits-style units string.
    data : ndarray or datetime.datetime
        Numeric data which in conjuction with the units define a set of
        udunit times that map directly to the input dates. If the input
        dates is not iterable, then the return value is a scalar;
        otherwise it is a numpy array of the same shape as dates.
    """
    iter_flag = hasattr(dates, '__iter__')
    if iter_flag:
        if not hasattr(dates, '__len__'):
            dates = list(dates)
        dates = np.asarray(dates)
        # coards.to_udunits will annoyingly return None when given
        # datetime.date as input rather than raising an exception, so
        # we have to un-pythonically test that the input is
        # datetime.datetime
        type_tester = np.vectorize(lambda d: isinstance(d, datetime.datetime))
        if not type_tester(dates).all():
            raise ValueError("dates must contain datetime instances (not date)")
        min_date = np.min(dates)
    else:
        # We assume that this is a scalar datetime
        if not isinstance(dates, datetime.datetime):
            raise ValueError("dates must be a datetime instance (not date)")
        min_date = dates
    if units is None:
        # Default units are days since the earliest date in the input
        units = min_date.strftime('days since %Y-%m-%d')
    if not isinstance(units, basestring):
        raise TypeError("units must be string")
    try:
        assert(coards.to_udunits(min_date, units) is not None)
    except:
        raise ValueError("units is not a valid udunits string")
    if iter_flag:
        converter = np.vectorize(lambda d: coards.to_udunits(d, units))
        data = converter(dates)
    else:
        data = coards.to_udunits(dates, units)
    return units, data