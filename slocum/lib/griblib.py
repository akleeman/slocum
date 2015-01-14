import os
import numpy as np
import logging
import netCDF4 as nc4

import slocum.lib.conventions as conv

from slocum.lib import units

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

try:
    # because gribapi is so difficult to install we make it optional.
    import gribapi
    _has_gribapi = True
except ImportError:
    logger.warn("gribapi is not installed, grib creation will not work.")
    _has_gribapi = False

_sample_file = os.path.join(os.path.dirname(__file__),
                        '../../../data/GFS20131226164503639.grb')

codes = {
0: ("Reserved",),
1: ("Pressure", "Pa"),
2: ("Pressure reduced to MSL", "Pa"),
3: ("Pressure tendency", "Pa s-1"),
4: ("Potential vorticity", "K m2 kg-1 s-1"),
5: ("ICAO Standard Atmosphere reference height", "m"),
6: ("Geopotential", "m2 s-2"),
7: ("Geopotential height", "gpm"),
8: ("Geometrical height", "m"),
9: ("Standard deviation of height", "m"),
10: ("Total ozone", "Dobson"),
11: ("Temperature", "K"),
12: ("Virtual temperature", "K"),
13: ("Potential temperature", "K"),
14: ("Pseudo-adiabatic potential temperature", "K"),
15: ("Maximum temperature", "K"),
16: ("Minimum temperature", "K"),
17: ("Dew-point temperature", "K"),
18: ("Dew-point depression (or deficit)", "K"),
19: ("Lapse rate", "K m-1"),
20: ("Visibility", "m"),
21: ("Radar spectra (1)", "-"),
22: ("Radar spectra (2)", "-"),
23: ("Radar spectra (3)", "-"),
24: ("Parcel lifted index (to 500 hPa) (see Note 6)", "K"),
25: ("Temperature anomaly", "K"),
26: ("Pressure anomaly", "Pa"),
27: ("Geopotential height anomaly", "gpm"),
28: ("Wave spectra (1)", "-"),
29: ("Wave spectra (2)", "-"),
30: ("Wave spectra (3)", "-"),
31: ("Wind direction", "Degree true"),
32: ("Wind speed", "m s-1"),
33: ("u-component of wind", "m/s"),
34: ("v-component of wind", "m/s"),
35: ("Stream function", "m2 s-1"),
36: ("Velocity potential", "m2 s-1"),
37: ("Montgomery stream function", "m2 s-1"),
38: ("Sigma coordinate vertical velocity", "s-1"),
39: ("Vertical velocity", "Pa s-1"),
40: ("Vertical velocity", "m s-1"),
41: ("Absolute vorticity", "s-1"),
42: ("Absolute divergence", "s-1"),
43: ("Relative vorticity", "s-1"),
44: ("Relative divergence", "s-1"),
45: ("Vertical u-component shear", "s-1"),
46: ("Vertical v-component shear", "s-1"),
47: ("Direction of current", "Degree true"),
48: ("Speed of current", "m s-1"),
49: ("u-component of current", "m s-1"),
50: ("v-component of current", "m s-1"),
51: ("Specific humidity", "kg kg-1"),
52: ("Relative humidity", "%"),
53: ("Humidity mixing ratio", "kg kg-1"),
54: ("Precipitable water", "kg m-2"),
55: ("Vapor pressure", "Pa"),
56: ("Saturation deficit", "Pa"),
57: ("Evaporation", "kg m-2"),
58: ("Cloud ice", "kg m-2"),
59: ("Precipitation rate", "kg m-2 s-1"),
60: ("Thunderstorm probability", "%"),
61: ("Total precipitation", "kg m-2"),
62: ("Large scale precipitation", "kg m-2"),
63: ("Convective precipitation", "kg m-2"),
64: ("Snowfall rate water equivalent", "kg m-2 s-1"),
65: ("Water equivalent of accumulated snow depth", "kg m-2"),
66: ("Snow depth", "m"),
67: ("Mixed layer depth", "m"),
68: ("Transient thermocline depth", "m"),
69: ("Main thermocline depth", "m"),
70: ("Main thermocline anomaly", "m"),
71: ("Total cloud cover", "%"),
72: ("Convective cloud cover", "%"),
73: ("Low cloud cover", "%"),
74: ("Medium cloud cover", "%"),
75: ("High cloud cover", "%"),
76: ("Cloud water", "kg m-2"),
77: ("Best lifted index (to 500 hPa) (see Note 6)", "K"),
78: ("Convective snow", "kg m-2"),
79: ("Large scale snow", "kg m-2"),
80: ("Water temperature", "K"),
81: ("Land cover (1 = land, 0 = sea)", "Proportion"),
82: ("Deviation of sea level from mean", "m"),
83: ("Surface roughness", "m"),
84: ("Albedo", "%"),
85: ("Soil temperature", "K"),
86: ("Soil moisture content", "kg m-2"),
87: ("Vegetation", "%"),
88: ("Salinity", "kg kg-1"),
89: ("Density", "kg m-3"),
90: ("Water run-off", "kg m-2"),
91: ("Ice cover (1 = ice, 0 = no ice)", "Proportion"),
92: ("Ice thickness", "m"),
93: ("Direction of ice drift", "Degree true"),
94: ("Speed of ice drift", "m s-1"),
95: ("u-component of ice drift", "m s-1"),
96: ("v-component of ice drift", "m s-1"),
97: ("Ice growth rate", "m s-1"),
98: ("Ice divergence", "s-1"),
99: ("Snow melt", "kg m-2"),
100: ("Significant height of combined wind waves and swell", "m"),
101: ("Direction of wind waves", "Degree true"),
102: ("Significant height of wind waves", "m"),
103: ("Mean period of wind waves", "s"),
104: ("Direction of swell waves", "Degree true"),
105: ("Significant height of swell waves", "m"),
106: ("Mean period of swell waves", "s"),
107: ("Primary wave direction", "Degree true"),
108: ("Primary wave mean period", "s"),
109: ("Secondary wave direction", "Degree true"),
110: ("Secondary wave mean period", "s"),
111: ("Net short-wave radiation flux (surface) (see Note 3)", "W m-2"),
112: ("Net long-wave radiation flux (surface) (see Note 3)", "W m-2"),
113: ("Net short-wave radiation flux (top of atmosphere) (see Note 3)", "W m-2"),
114: ("Net long-wave radiation flux (top of atmosphere) (see Note 3)", "W m-2"),
115: ("Long-wave radiation flux (see Note 3)", "W m-2"),
116: ("Short-wave radiation flux (see Note 3)", "W m-2"),
117: ("Global radiation flux (see Note 3)", "W m-2"),
118: ("Brightness temperature", "K"),
119: ("Radiance (with respect to wave number)", "W m-1 sr-1"),
120: ("Radiance (with respect to wave length)", "W m-3 sr-1"),
121: ("Latent heat flux", "W m-2"),
122: ("Sensible heat flux", "W m-2"),
123: ("Boundary layer dissipation", "W m-2"),
124: ("Momentum flux, u-component", "N m-2"),
125: ("Momentum flux, v-component", "N m-2"),
126: ("Wind mixing energy", "J"),
}

reverse_codes = dict((v[0], k) for k, v in codes.iteritems())

#http://www.nco.ncep.noaa.gov/pmb/docs/on388/table3.html
indicator_of_level = {"u-component of wind": 105,
                      "v-component of wind": 105}

level = {"u-component of wind": 10,
         "v-component of wind": 10}

_sec_per_hour = 3600


def set_time(source, grib):
    """
    Sets the dataDate, dataTime, unitOfTimeRange, P2, timeRangeIndicator,
    parameters in the grib message 'grib' using the time variable in source.
    """
    if source[conv.TIME].size != 1:
        raise ValueError("expected a single time step")
    # analysis, forecast start, verify time, obs time,
    # (start of forecast for now)
    unit = source[conv.TIME].attrs[conv.UNITS]
    # reference time is assumed to be the origin of the source
    # time variable.  This is the case with GFS but perhaps
    # not with other forecasts.
    rt = nc4.num2date([0], unit)[0]
    gribapi.grib_set_long(grib, "dataDate", "%04d%02d%02d" % (rt.year,
                                                              rt.month,
                                                              rt.day))
    gribapi.grib_set_long(grib, "dataTime", "%02d%02d" % (rt.hour, rt.minute))
    # taken from ECMWF grib tables
    unit_codes = {'minute': 0,
             'hour': 1,
             'day': 2,
             'month': 3,
             'year': 4,
             'second': 254}
    grib_time_code = None
    for k, v in unit_codes.iteritems():
        if unit.lower().startswith(k):
            grib_time_code = v
    if grib_time_code is None:
        raise ValueError("Unexpected unit")
    gribapi.grib_set_long(grib, 'unitOfTimeRange', 1)
    vt = np.asscalar(source[conv.TIME].values)
    assert int(vt) == vt
    gribapi.grib_set_long(grib, 'P2', vt)
    # forecast is valid at reference + P2
    gribapi.grib_set_long(grib, "timeRangeIndicator", 10)


def set_grid(source, grib):
    """
    Infers the grid from source's longitude and latitude varibles and
    writes grib message parameters:

    gridDefinition, gridDescriptionSectionPresent,  shapeOfTheEarth,
    Ni, Nj, latitudeOfFirstGridPoint, latitudeOfLastGridPoint,
    longitudeOfFirstGridPoint, longitudeOfLastGridPoint.
    """
    # define the grid
    gribapi.grib_set_long(grib, "gridDefinition", 255)
    gribapi.grib_set_long(grib, "gridDescriptionSectionPresent", 1)

    gribapi.grib_set_long(grib, "shapeOfTheEarth", 6)

    dim_shape = dict(zip(source.dimensions, source.shape))
    gribapi.grib_set_long(grib, "Ni", dim_shape[conv.LON])
    gribapi.grib_set_long(grib, "Nj", dim_shape[conv.LAT])

    lon = source[conv.LON].values
    lat = source[conv.LAT].values
    # TODO: Dateline but: this will break when crossing the dateline.
    assert np.unique(np.diff(lon)).size == 1
    assert np.unique(np.diff(lat)).size == 1
    assert lon.ndim == 1
    assert lat.ndim == 1
    gribapi.grib_set_long(grib, "jScansPositively", 1)
    gribapi.grib_set_long(grib, "latitudeOfFirstGridPoint",
                          int(np.min(lat) * 1000))
    gribapi.grib_set_long(grib, "latitudeOfLastGridPoint",
                          int(np.max(lat) * 1000))
    gribapi.grib_set_long(grib, "iScansPositively", 1)
    gribapi.grib_set_long(grib, "longitudeOfFirstGridPoint",
                          int(lon[0] * 1000))
    gribapi.grib_set_long(grib, "longitudeOfLastGridPoint",
                          int(lon[-1] * 1000))


def get_varible_name(source):
    """
    Infers the variable name of source by assuming there is only one
    non-coordinate.
    """
    return source.name


def set_product(source, grib):
    """
    Sets the 'inidcatorOfParameter', 'table2Version', 'indicatorOfTypeOfLevel'
    and 'level' parameters in a grib message by inferring their values from
    the only non-coordinate in source.
    """
    var_name = get_varible_name(source)
    grib_var_name = conv.to_grib1[var_name]
    gribapi.grib_set_long(grib, 'indicatorOfParameter',
                          reverse_codes[grib_var_name])
    gribapi.grib_set_long(grib, 'table2Version', 2)

    gribapi.grib_set_long(grib, 'indicatorOfTypeOfLevel',
                          indicator_of_level[grib_var_name])
    gribapi.grib_set_long(grib, 'level',
                          level[grib_var_name])


def set_data(source, grib):
    """
    Sets the actual data of a grib message.
    """
    var_name = get_varible_name(source)
    # treat masked arrays differently
    if isinstance(source[var_name].values, np.ma.core.MaskedArray):
        gribapi.grib_set(grib, "bitmapPresent", 1)
        # use the missing value from the masked array as default
        missing_value = source[var_name].values.get_fill_value()
        # but give the netCDF specified missing value preference
        missing_value = source[var_name].attrs.get('missing_value',
                                                        missing_value)
        gribapi.grib_set_double(grib, "missingValue",
                                float(missing_value))
        data = source[var_name].values.filled()
    else:
        gribapi.grib_set_double(grib, "missingValue", 9999)
        data = source[var_name].values[:]
    gribapi.grib_set_long(grib, "bitsPerValue", 12)
    #gribapi.grib_set_long(grib, "bitsPerValueAndRepack", 12)
    gribapi.grib_set_long(grib, "decimalPrecision", 2)
    gribapi.grib_set_long(grib, "decimalScaleFactor", 2)
    #gribapi.grib_set_long(grib, "binaryScaleFactor", 0)
    gribapi.grib_set_long(grib, "dataRepresentationType", 0)
    # get the grib code for the variable
    code = reverse_codes[conv.to_grib1[var_name]]
    _, grib_unit = codes[code]
    # default to the grib default unit
    unit = source[var_name].attrs.get(conv.UNITS, grib_unit)
    mult = 1.
    if not unit == grib_unit:
        mult = units._speed[unit] / units._speed[grib_unit]
    # add the data
    gribapi.grib_set_double_array(grib, "values", mult * data.flatten())


def save(source, target, append=False, sample_file=_sample_file):
    """
    Takes a dataset (source) and writes its contents
    as grib 1 to file-like target.  Grib 1 is used (instead
    of grib 2) because some older forecast visualization
    software can't read grib 2.

    This is a heavily modified but none-the-less derivative of
    the grib saving functions from the iris package.

    Parameters
    ----------
    source : Dataset
        A netcdf-like file holding the dataset we want to write
        as grib.  This must contain time, longitude and latitude
        coordinates in order to infer the grib grid and time params
    target : string path or file-like
        Where the contents should be written.  If target is a string
        the file is created or appended to.
    append : boolean
        When creating a new file from string you can optionally
        append to the file.
    """
    if not _has_gribapi:
        raise ImportError("gripapi is required to write grib files.")

    if isinstance(target, basestring):
        grib_file = open(target, "ab" if append else "wb")
    elif hasattr(target, "write"):
        if hasattr(target, "mode") and "b" not in target.mode:
            raise ValueError("Target not binary")
        grib_file = target
    else:
        raise ValueError("Can only save grib to filename or writable")
    if not conv.LAT in source.variables or not conv.LON in source.variables:
        raise ValueError("Did not find either latitude or longitude.")
    if source[conv.LAT].ndim != 1 or source[conv.LON].ndim != 1:
        raise ValueError("Latitude and Longitude should be regular.")
    if not conv.TIME in source.variables:
        raise ValueError("Expected time coordinate")
    # sort the lats and lons
    source = source.indexed(latitude=np.argsort(source[conv.LAT].values))
    lons = source[conv.LON].values
    if np.any(np.abs(np.diff(lons)) > 180.):
        # the latitudes must cross the dateline since we only allow 180
        # degree wide bounding boxes, and there is more than a 180 degree
        # difference between longitudes.  Instead we try converting to
        # 0 to 360 degree longitudes before sorting.
        lons = np.mod(lons, 360)
        if np.any(np.abs(np.diff(lons)) > 180.):
            # TODO: I'm sure theres a way to deal with arbitrary longitude
            # specifications for global data ... but its not a high priority
            # so that will wait for later.
            raise ValueError("Longitudes span more than 180 degrees and the dateline?")
    source[conv.LON].values[:] = lons
    source = source.indexed(longitude=np.argsort(lons))
    # iterate over variables, unless they are considered
    # auxiliary variables (ie, variables used by slocum
    # but not in grib files).
    auxilary_variables = [conv.WIND_SPEED, conv.WIND_DIR]
    for single_var in (v for k, v in source.noncoordinates.iteritems()
                       if not k in auxilary_variables):
        # then iterate over time slices
        iter_time = (single_var.indexed(**{conv.TIME: [i]})
                     for i in range(single_var.coordinates[conv.TIME].size))
        for obj in iter_time:
            # Save this slice to the grib file
            gribapi.grib_gribex_mode_off()
            if sample_file is not None and os.path.exists(sample_file):
                with open(sample_file, 'r') as f:
                    grib_message = gribapi.grib_new_from_file(f)
                logger.info("Created grib message from file %s" % sample_file)
            else:
                logger.info("Creating grib message from gribapi sample: GRIB1")
                grib_message = gribapi.grib_new_from_samples("GRIB1")
            set_time(obj, grib_message)
            set_product(obj, grib_message)
            set_grid(obj, grib_message)
            set_data(obj, grib_message)
            gribapi.grib_write(grib_message, grib_file)
            gribapi.grib_release(grib_message)
    # if target was a string then we have to close the file we
    # created, otherwise leave that up to the user.
    if isinstance(target, basestring):
        grib_file.close()