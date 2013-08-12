import os
import grib2
import numpy as np
import coards
import pygrib
import itertools

from cStringIO import StringIO
from datetime import datetime

import sl.objects.conventions as conv

from sl.lib import datelib
from sl.objects import core, units

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
33 : ("u-component of wind (see Note 4)", "m s-1"),
34 : ("v-component of wind (see Note 4)", "m s-1"),
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
45 : ("Vertical u-component shear (see Note 4)", "s-1"),
46 : ("Vertical v-component shear (see Note 4)", "s-1"),
47 : ("Direction of current", "Degree true"),
48 : ("Speed of current", "m s-1"),
49 : ("u-component of current (see Note 4)", "m s-1"),
50 : ("v-component of current (see Note 4)", "m s-1"),
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
95 : ("u-component of ice drift (see Note 4)", "m s-1"),
96 : ("v-component of ice drift (see Note 4)", "m s-1"),
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
124 : ("Momentum flux, u-component (see Note 4)", "N m-2"),
125 : ("Momentum flux, v-component (see Note 4)", "N m-2"),
126 : ("Wind mixing energy", "J"),
}

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
            udunits = coards.to_udunits(start_date,
                                        'hours since %Y-%m-%d %H:%M:%S')
            uddates = [coards.datetime_to_udunits(d, udunits) for d in dates]
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
                                      attributes={conv.UNITS:udunits})

            obj.create_variable(var, dim=(conv.TIME, conv.LAT, conv.LON), data=data)
        neg_lon = obj[conv.LON].data <= 0
        obj[conv.LON].data[neg_lon] = 360. + obj[conv.LON].data[neg_lon]
        if 'unknown' in obj.variables:
            obj.delete_variable('unknown')
        yield units.normalize_data(obj)

def grib_ready(nc, gribs):
    """
    (Pdb) print nc['lon']
      standard_name   | longitude                                         
      long_name       | longitude                                         
      units           | degrees_east                                      
      axis            | X         
      
      (Pdb) print nc['lat']
      standard_name   | latitude                                          
      long_name       | latitude                                          
      units           | degrees_north                                     
      axis            | Y                                                 
    """

    out = core.Data()
    out.attributes['Conventions'] = "CF-1.4"

    if 'ensemble' in nc:
        nc = nc.take([0], 'ensemble')
        nc = nc.renamed({'ensemble':'height'})

    if 'lon' in nc:
        attributes = {'long_name': 'longitude',
                      'units': 'degrees_east',
                      'standard_name': 'longitude',
                      'axis': 'X'}
        out.create_coordinate('lon', data=nc['lon'].data.astype('float64'),
                              attributes=attributes)
    if 'lat' in nc:
        attributes = {'long_name': 'latitude',
                      'units': 'degrees_north',
                      'standard_name': 'latitude',
                      'axis': 'Y'}
        out.create_coordinate('lat', data=nc['lat'].data.astype('float64'),
                              attributes=attributes)

    out.create_coordinate('height', data=[10.], attributes={'standard_name': 'height',
                                                       'long_name': 'height',
                                                       'units': 'm',
                                                       'positive': 'up',
                                                       'axis': 'Z'})

    if 'time' in nc:
        attributes = {'long_name': 'time',
                      'units': nc['time'].attributes['units'],
                      'standard_name': 'time',
                      'calendar': 'proleptic_gregorian'}
        zeros = np.zeros(nc['time'].data.shape)
        out.create_coordinate('time', data=zeros.astype('float64'),
                              attributes=attributes, record=True)

    dims = tuple(['time', 'height', 'lat', 'lon'])
    if 'uwnd' in nc:
        attributes = {'long_name': '10 metre U wind component',
                      'units': 'm s**-1',
                      'code': [33],
                      'table': [2]}
        order = tuple([nc['uwnd'].dimensions.index(d) for d in dims])
        data = nc['uwnd'].data.transpose(order)
        out.create_variable('10u', dim=dims,
                            data=data.astype('float32'), attributes=attributes)

    if 'vwnd' in nc:
        attributes = {'long_name': '10 metre V wind component',
                      'units': 'm s**-1',
                      'code': [34],
                      'table': [2]}
        order = tuple([nc['vwnd'].dimensions.index(d) for d in dims])
        data = nc['vwnd'].data.transpose(order)
        out.create_variable('10v', dim=dims,
                            data=data.astype('float32'), attributes=attributes)

    with open(os.path.join(os.path.dirname(__file__), '../../../data/repaired.nc'), 'w') as f:
        out.dump(f)
    import pdb; pdb.set_trace()

def change_time(fn, reference):
    grbs = pygrib.open(fn)

    var_names = {'10 metre U wind component':'uwnd',
                 '10 metre V wind component':'vwnd'}
    out_fn = os.path.join(os.path.dirname(__file__), '../../../data/time_change.grb')
    out = open(out_fn, 'wb')
    for msg in grbs:
        var = var_names[msg.name]
        data = reference[var].data[0, :]
        norms = [np.linalg.norm(msg.values - x, 'fro') for x in data]
        i = np.argmin(norms)
        step_range = unicode(int(reference[conv.TIME].data[i]))
        msg.__setattr__('stepRange', step_range)
#        msg.__setattr__('dataDate', int(date.strftime('%Y%m%d')))
#        msg.__setattr__('dataTime', int(date.strftime('%H%M')))
        out.write(msg.tostring())

    out.close()
    new_grbs = pygrib.open(out_fn)
    import pdb; pdb.set_trace()

if __name__ == "__main__":


    fn = os.path.join(os.path.dirname(__file__), '../../../data/windbreaker.nc')
    with open(fn) as f:
        nc = core.Data(f.read())
    change_time(os.path.join(os.path.dirname(__file__), '../../../data/repaired.grb'),
                reference=nc)

    # gribs = grib2.Grib2Decode(fn)
    gribs = pygrib.open(fn)
    grib_ready(nc, gribs)

    obj = list(degrib(fn))[0]
    gribs = pygrib.open(fn)
    import pdb; pdb.set_trace()
    for obj in objs:
        obj.to_file(open('/home/kleeman/Desktop/tmp.nc', 'w'))
