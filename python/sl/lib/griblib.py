import os
import numpy as np
import coards
import pygrib
import gribapi
import netCDF4 as nc4
import itertools

from cStringIO import StringIO
from datetime import datetime

import sl.lib.conventions as conv
from sl.objects import units

codes = {
0 : ("Reserved",),
1 : ("Pressure", "Pa"),
2 : ("Pressure reduced to MSL", "Pa"),
3 : ("Pressure tendency", "Pa s-1"),
4 : ("Potential vorticity", "K m2 kg-1 s-1"),
5 : ("ICAO Standard Atmosphere reference height", "m"),
6 : ("Geopotential", "m2 s-2"),
7 : ("Geopotential height", "gpm"),
8 : ("Geometrical height", "m"),
9 : ("Standard deviation of height", "m"),
10 : ("Total ozone", "Dobson"),
11 : ("Temperature", "K"),
12 : ("Virtual temperature", "K"),
13 : ("Potential temperature", "K"),
14 : ("Pseudo-adiabatic potential temperature", "K"),
15 : ("Maximum temperature", "K"),
16 : ("Minimum temperature", "K"),
17 : ("Dew-point temperature", "K"),
18 : ("Dew-point depression (or deficit)", "K"),
19 : ("Lapse rate", "K m-1"),
20 : ("Visibility", "m"),
21 : ("Radar spectra (1)", "-"),
22 : ("Radar spectra (2)", "-"),
23 : ("Radar spectra (3)", "-"),
24 : ("Parcel lifted index (to 500 hPa) (see Note 6)", "K"),
25 : ("Temperature anomaly", "K"),
26 : ("Pressure anomaly", "Pa"),
27 : ("Geopotential height anomaly", "gpm"),
28 : ("Wave spectra (1)", "-"),
29 : ("Wave spectra (2)", "-"),
30 : ("Wave spectra (3)", "-"),
31 : ("Wind direction", "Degree true"),
32 : ("Wind speed", "m s-1"),
33 : ("u-component of wind", "m/s"),
34 : ("v-component of wind", "m/s"),
35 : ("Stream function", "m2 s-1"),
36 : ("Velocity potential", "m2 s-1"),
37 : ("Montgomery stream function", "m2 s-1"),
38 : ("Sigma coordinate vertical velocity", "s-1"),
39 : ("Vertical velocity", "Pa s-1"),
40 : ("Vertical velocity", "m s-1"),
41 : ("Absolute vorticity", "s-1"),
42 : ("Absolute divergence", "s-1"),
43 : ("Relative vorticity", "s-1"),
44 : ("Relative divergence", "s-1"),
45 : ("Vertical u-component shear", "s-1"),
46 : ("Vertical v-component shear", "s-1"),
47 : ("Direction of current", "Degree true"),
48 : ("Speed of current", "m s-1"),
49 : ("u-component of current", "m s-1"),
50 : ("v-component of current", "m s-1"),
51 : ("Specific humidity", "kg kg-1"),
52 : ("Relative humidity", "%"),
53 : ("Humidity mixing ratio", "kg kg-1"),
54 : ("Precipitable water", "kg m-2"),
55 : ("Vapor pressure", "Pa"),
56 : ("Saturation deficit", "Pa"),
57 : ("Evaporation", "kg m-2"),
58 : ("Cloud ice", "kg m-2"),
59 : ("Precipitation rate", "kg m-2 s-1"),
60 : ("Thunderstorm probability", "%"),
61 : ("Total precipitation", "kg m-2"),
62 : ("Large scale precipitation", "kg m-2"),
63 : ("Convective precipitation", "kg m-2"),
64 : ("Snowfall rate water equivalent", "kg m-2 s-1"),
65 : ("Water equivalent of accumulated snow depth", "kg m-2"),
66 : ("Snow depth", "m"),
67 : ("Mixed layer depth", "m"),
68 : ("Transient thermocline depth", "m"),
69 : ("Main thermocline depth", "m"),
70 : ("Main thermocline anomaly", "m"),
71 : ("Total cloud cover", "%"),
72 : ("Convective cloud cover", "%"),
73 : ("Low cloud cover", "%"),
74 : ("Medium cloud cover", "%"),
75 : ("High cloud cover", "%"),
76 : ("Cloud water", "kg m-2"),
77 : ("Best lifted index (to 500 hPa) (see Note 6)", "K"),
78 : ("Convective snow", "kg m-2"),
79 : ("Large scale snow", "kg m-2"),
80 : ("Water temperature", "K"),
81 : ("Land cover (1 = land, 0 = sea)", "Proportion"),
82 : ("Deviation of sea level from mean", "m"),
83 : ("Surface roughness", "m"),
84 : ("Albedo", "%"),
85 : ("Soil temperature", "K"),
86 : ("Soil moisture content", "kg m-2"),
87 : ("Vegetation", "%"),
88 : ("Salinity", "kg kg-1"),
89 : ("Density", "kg m-3"),
90 : ("Water run-off", "kg m-2"),
91 : ("Ice cover (1 = ice, 0 = no ice)", "Proportion"),
92 : ("Ice thickness", "m"),
93 : ("Direction of ice drift", "Degree true"),
94 : ("Speed of ice drift", "m s-1"),
95 : ("u-component of ice drift", "m s-1"),
96 : ("v-component of ice drift", "m s-1"),
97 : ("Ice growth rate", "m s-1"),
98 : ("Ice divergence", "s-1"),
99 : ("Snow melt", "kg m-2"),
100 : ("Significant height of combined wind waves and swell", "m"),
101 : ("Direction of wind waves", "Degree true"),
102 : ("Significant height of wind waves", "m"),
103 : ("Mean period of wind waves", "s"),
104 : ("Direction of swell waves", "Degree true"),
105 : ("Significant height of swell waves", "m"),
106 : ("Mean period of swell waves", "s"),
107 : ("Primary wave direction", "Degree true"),
108 : ("Primary wave mean period", "s"),
109 : ("Secondary wave direction", "Degree true"),
110 : ("Secondary wave mean period", "s"),
111 : ("Net short-wave radiation flux (surface) (see Note 3)", "W m-2"),
112 : ("Net long-wave radiation flux (surface) (see Note 3)", "W m-2"),
113 : ("Net short-wave radiation flux (top of atmosphere) (see Note 3)", "W m-2"),
114 : ("Net long-wave radiation flux (top of atmosphere) (see Note 3)", "W m-2"),
115 : ("Long-wave radiation flux (see Note 3)", "W m-2"),
116 : ("Short-wave radiation flux (see Note 3)", "W m-2"),
117 : ("Global radiation flux (see Note 3)", "W m-2"),
118 : ("Brightness temperature", "K"),
119 : ("Radiance (with respect to wave number)", "W m-1 sr-1"),
120 : ("Radiance (with respect to wave length)", "W m-3 sr-1"),
121 : ("Latent heat flux", "W m-2"),
122 : ("Sensible heat flux", "W m-2"),
123 : ("Boundary layer dissipation", "W m-2"),
124 : ("Momentum flux, u-component", "N m-2"),
125 : ("Momentum flux, v-component", "N m-2"),
126 : ("Wind mixing energy", "J"),
}

reverse_codes = dict((v[0], k) for k, v in codes.iteritems())

#http://www.nco.ncep.noaa.gov/pmb/docs/on388/table3.html
indicator_of_level = {"u-component of wind" : 105,
                      "v-component of wind" : 105}

level = {"u-component of wind" : 10,
         "v-component of wind" : 10}

_sec_per_hour = 3600

def degrib(fn):
    """
    Takes a sequence of grib messages and converts them into a sequence of
    data objects grouped such that each data object contains all variables that
    share lat long grids.

    It is assumed that all messages for each group of grids have the same times.
    If this is not the case an exception is thrown
    """
    if not os.path.exists(fn):
        raise ValueError("grib file %s does not exist" % fn)

    gribs = pygrib.open(fn)
    def get_grid(x):
        # extracts the stringified lat long variables from a message
        return "%s\t%s" % (x.distinctLatitudes.tostring(),
                           x.distinctLongitudes.tostring())
    # the actual order doesn't matter we just want to make sure they're grouped
    gribs = sorted(gribs, key=get_grid)
    for grid, group in itertools.groupby(gribs, key=get_grid):
        # create a new object for each grid
        obj = core.Data()
        lats, lons = [np.fromstring(x) for x in grid.split('\t')]
        obj.create_coordinate(conv.LAT, lats)
        obj.create_coordinate(conv.LON, lons)

        var = lambda x: x.name
        for var, var_group in itertools.groupby(group, key=var):
            # iterate over all messages with the same variable name, turning
            # each var into core.variable
            def iterate():
                # processes each message returning the date and values
                g = var_group.next()
                for g in itertools.chain([g], var_group):
                    assert g.unitOfTimeRange == 1
                    pre_format = '%s %.4d' % (g.validityDate, float(g.validityTime))
                    valid_time = datetime.strptime(pre_format, '%Y%m%d %H%M')
                    yield valid_time, g.values

            # extract all the dates so we can make the time coordinate
            iter_times = list(iterate())
            dates = [x[0] for x in iter_times]
            start_date = min(dates)
            udunit = coards.to_udunits(start_date,
                                        'hours since %Y-%m-%d %H:%M:%S')
            uddates = [coards.datetime_to_udunits(d, udunit) for d in dates]
            # create an empty data object and fill it
            data = np.zeros((len(uddates), lats.size, lons.size))
            for i, (_, x) in enumerate(iter_times):
                data[i, :, :] = x
            # if the time coordinate exists make sure it matches
            if conv.TIME in obj.variables:
                if not np.all(uddates == obj[conv.TIME].data):
                    # overlap_dates = sorted(set(uddates).intersection(obj[conv.TIME].data))
                    raise ValueError("expected all time variables to match")
            else:
                obj.create_coordinate(conv.TIME, uddates, record=True,
                                      attributes={conv.UNITS:udunit})

            obj.create_variable(var, dim=(conv.TIME, conv.LAT, conv.LON), data=data)
        neg_lon = obj[conv.LON].data <= 0
        obj[conv.LON].data[neg_lon] = 360. + obj[conv.LON].data[neg_lon]
        if 'unknown' in obj.variables:
            obj.delete_variable('unknown')
        yield units.normalize_data(obj)

def set_time(source, grib):
    if source[conv.TIME].size != 1:
        raise ValueError("expected a single time step")
    # analysis, forecast start, verify time, obs time,
    # (start of forecast for now)
    unit = source[conv.TIME].attributes[conv.UNITS]
    # reference time is assumed to be the reference to the unit.
    rt = nc4.num2date([0], unit)[0]
    gribapi.grib_set_long(grib, "dataDate", "%04d%02d%02d" % (rt.year,
                                                              rt.month,
                                                              rt.day))
    gribapi.grib_set_long(grib, "dataTime", "%02d%02d" % (rt.hour, rt.minute))

    unit_codes = {'minute' : 0,
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

    vt = np.asscalar(source[conv.TIME][:])
    assert int(vt) == vt
    gribapi.grib_set_long(grib, 'P2', vt)
    # forecast is valid at reference + tp
    gribapi.grib_set_long(grib, "timeRangeIndicator", 10)

def set_grid(source, grib):
    # define the grid
    gribapi.grib_set_long(grib, "gridDefinition", 255)
    gribapi.grib_set_long(grib, "gridDescriptionSectionPresent", 1)

    gribapi.grib_set_long(grib, "shapeOfTheEarth", 6)

    gribapi.grib_set_long(grib, "Ni", source.dimensions[conv.LON])
    gribapi.grib_set_long(grib, "Nj", source.dimensions[conv.LAT])

    lat = source[conv.LAT]
    lon = source[conv.LON]
    gribapi.grib_set_long(grib, "latitudeOfFirstGridPoint",
                          int(lat[0]*1000))
    gribapi.grib_set_long(grib, "latitudeOfLastGridPoint",
                          int(lat[-1]*1000))
    gribapi.grib_set_long(grib, "longitudeOfFirstGridPoint",
                          int((lon[0] % 360)*1000))
    gribapi.grib_set_long(grib, "longitudeOfLastGridPoint",
                          int((lon[-1] % 360)*1000))

def get_varible_name(source):
    variables = source.noncoordinates
    if not len(variables) == 1:
        raise ValueError("expected a single variable")
    return variables.keys()[0]

def set_product(source, grib):
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
    var_name = get_varible_name(source)
    # treat masked arrays differently
    if isinstance(source[var_name].data, np.ma.core.MaskedArray):
        gribapi.grib_set(grib, "bitmapPresent", 1)
        missing_value = source[var_name].attributes.get('missing_value', 9999)
        gribapi.grib_set_double(grib, "missingValue",
                                float(missing_value))
        data = source[var_name].data.filled()
    else:
        gribapi.grib_set_double(grib, "missingValue", 9999)
        data = source[var_name].data[:]

    gribapi.grib_set_long(grib, "dataRepresentationType", 0)

    code = reverse_codes[conv.to_grib1[var_name]]
    _, grib_unit = codes[code]
    # default to the grib default unit
    unit = source[var_name].attributes.get(conv.UNITS, grib_unit)
    mult = 1.
    if not unit == grib_unit:
        mult = units._speed[unit] / units._speed[grib_unit]
    # add the data
    gribapi.grib_set_double_array(grib, "values", mult * data.flatten())

def save(source, target, append=False):
    # grib file (this bit is common to the pp and grib savers...)
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

    for v in source.noncoordinates.keys():
        single_var = source.select([v])
        for t, obj in single_var.iterator(conv.TIME):
            # Save this slice to the grib file
            grib_message = gribapi.grib_new_from_samples("GRIB1")
            set_time(obj, grib_message)
            set_product(obj, grib_message)
            set_grid(obj, grib_message)
            set_data(obj, grib_message)
            gribapi.grib_write(grib_message, grib_file)
            gribapi.grib_release(grib_message)

    # (this bit is common to the pp and grib savers...)
    if isinstance(target, basestring):
        grib_file.close()