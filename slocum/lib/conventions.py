import xray
import coards
import numpy as np

from datetime import datetime

try:
    import netCDF4 as nc4
    _has_nc4 = True
except:
    _has_nc4 = False

ENSEMBLE = 'ens'
ENS_SPREAD_WS = 'ws_spread'
TIME = 'time'
HOURS = 'hours'
LAT = 'latitude'
LON = 'longitude'
UWND = 'uwnd'
VWND = 'vwnd'
PRECIP = 'precip'
CLOUD = 'cloud_cover'
PRESSURE = 'pressure'
UNITS = 'units'
BEARING = 'bearing'
HEADING = 'heading'
SPEED = 'speed'
DISTANCE = 'distance'
RELATIVE_WIND = 'rel_wind'
MOTOR_ON = 'motor_on'
WIND = 'wind'
WIND_SPEED = 'wind_speed'
WIND_DIR = 'wind_dir'
STEP = 'step'
NUM_STEPS = 'num_steps'

to_grib1 = {UWND: "u-component of wind",
            VWND: "v-component of wind"}


def _clean_cf_time_units(units):
    delta, ref_date = xray.conventions._unpack_netcdf_time_units(units)
    delta = delta.lower()
    if not delta.endswith('s'):
        delta = '%ss' % delta
    return '%s since %s' % (delta, ref_date)


def _encode_cf_datetime_using_coards(dates, units=None, calendar=None):
    dates = np.asarray(dates)

    if units is None:
        units = xray.conventions.infer_datetime_units(dates)
    else:
        units = xray.conventions._cleanup_netcdf_time_units(units)

    units = _clean_cf_time_units(units)

    if calendar is None:
        calendar = 'standard'
    assert calendar in xray.conventions._STANDARD_CALENDARS

    if np.issubdtype(dates.dtype, np.datetime64):
        # for now, don't bother doing any trickery like decode_cf_datetime to
        # convert dates to numbers faster
        # note: numpy's broken datetime conversion only works for us precision
        dates = dates.astype('M8[us]').astype(datetime)
    # coards only works with standard calendar

    def encode_datetime(d):
        return np.nan if d is None else coards.to_udunits(d, units)

    num = np.array([encode_datetime(d) for d in dates.flat])
    num = num.reshape(dates.shape)
    return (num, units, calendar)

encode_cf_datetime = (xray.conventions.encode_cf_datetime
                      if _has_nc4 else _encode_cf_datetime_using_coards)


def encode_cf_time_variable(time_var):
    dates = time_var.values
    units = time_var.encoding.pop('units',
                                  time_var.attrs.pop('units', None))
    calendar = time_var.encoding.pop('calendar',
                                     time_var.attrs.pop('calendar', None))
    values, units, calendar = encode_cf_datetime(dates, units, calendar)
    attrs = time_var.attrs.copy()
    attrs.update({'units': units,
                  'calendar': calendar})
    return xray.Variable(time_var.dims,
                         values, attrs, time_var.encoding)


def decode_cf_datetime(num_dates, units, calendar=None):
    if not _has_nc4:
        calendar = calendar or 'standard'
        assert calendar in xray.conventions._STANDARD_CALENDARS
    return xray.conventions.decode_cf_datetime(num_dates, units, calendar)