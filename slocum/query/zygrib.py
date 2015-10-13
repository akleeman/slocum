# WIND    :    wind at 10m
# GUST    :    wind gust
# PRESS    :    MSL pressure
# CAPE    :    CAPE (convective available potential energy)
# TEMP    :    temperature at 2m
# TMIN    :    temperature min at 2m
# TMAX    :    temperature max at 2m
# PRECIP    :    precipitation
# CLOUD    :    cloud cover
# HREL    :    relative humidity
# ISOTH0    :    isotherm 0 C
# SNOWD    :    snow depth
# SNOWC    :    snow risk
# FRZRAIN    :    frozen rain risk
# A200    :    data in altitude (200hPa = 11800m)
# geopotential altitude, wind, temperature, theta-e, relative humidity
# A300    :    data in altitude (300hPa = 9200m)
# A500    :    data in altitude (500hPa = 5600m)
# A700    :    data in altitude (700hPa = 3000m)
# A850    :    data in altitude (850hPa = 1460m)
# A925    :    data in altitude (925hPa = 760m)
# WVSIG    :    significant waves height
# WVMAX    :    maximum waves
# WVSWEL    :    swell
# WVWIND    :    wind waves
# WVWCAP    :    whitecap probability
# WVPRIM    :    primary waves
# WVSCDY    :    secondary waves

import re
import warnings
import numpy as np

import utils

_regex_divider = re.compile('\s*[:=;]\s*')

aliases = {'area': 'domain',
           'resol': 'resolution',
           'grid': 'resolution',
           }

parsers = {'domain': utils.parse_domain,
           'resolution': utils.parse_resolution,
           'days': int,
           'hours': int,
           }


def parse_query(query_str):
    lines = re.split('[\r\n]', query_str)
    # remove empty lines
    lines = filter(len, lines)
    lines = [_regex_divider.split(x, 1) for x in lines]
    keyvals = [kv for kv in lines if len(kv) == 2]

    def parse_pair(k, v):
        parser_pair = utils.lookup(k, parsers, aliases=aliases)
        if parser_pair is None:
            warnings.warn("ignoring field %s:%s" % (k, v))
            return None
        k, parser = parser_pair
        return k, parser(v)
    query = dict(filter(None, (parse_pair(k, v) for k, v in keyvals)))

    # assemble all the lines without :=; in them into a single
    # string of variables
    variables = ' '.join([kv[0] for kv in lines if len(kv) == 1])
    variables = re.split('\W*', variables)
    query['variables'] = variables
    return normalize(query)


def normalize(zygrib_query):
    # turn days/hours into hours
    hour_resolution = zygrib_query.pop('hours')
    days = zygrib_query.pop('days')
    hours = np.arange(0., days * 24. + hour_resolution, hour_resolution)
    zygrib_query['hours'] = hours
    zygrib_query['type'] = 'gridded'
    model = zygrib_query.get('model', None)
    zygrib_query['model'] = utils.validate_model(model)
    variables = utils.validate_variables(zygrib_query.get('variables', []))
    zygrib_query['variables'] = variables
    return zygrib_query
