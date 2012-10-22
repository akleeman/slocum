"""
Poseidon - The god of the sea.

This contains tools for accessing data about the future/past state of the seas.
"""
from __future__ import with_statement

import os
import re
import uuid
import numpy as np
import hashlib
import logging
import urllib
import urllib2
import datetime
import urlparse

from BeautifulSoup import BeautifulSoup

import wx.objects.conventions as conv
from wx.objects import objects, units, core

_data_dir = os.path.join(os.path.dirname(__file__), '../../data/')
_sources = {'ccmp_daily':'ccmp/mean_wind_%Y%m%d_v11l30flk.nc',
            'gefs': 'http://motherlode.ucar.edu:9080/thredds/ncss/grid/NCEP/GEFS/Global_1p0deg_Ensemble/member/GEFS_Global_1p0deg_Ensemble_%Y%m%d_0600.grib2',
            'nww3': 'http://motherlode.ucar.edu:9080/thredds/ncss/grid/fmrc/NCEP/WW3/Global/files/WW3_Global_20110103_1800.grib2/dataset.html',
}
_ibtracs = 'ibtracs/Allstorms.ibtracs_wmo.v03r02.nc'
_storms = 'historical_storms.nc'

def ncdf_subset(url, ll, ur, vars, dir=None, path=None):
    """
    Finds the latest forecast on a netcdf subset server
    """
    ur, ll = ensure_corners(ur, ll)
    query = {'var':','.join(vars.keys()),
             'north':'%.2f' % np.ceil(ur.lat),
             'west':'%.2f' % np.floor(ll.lon),
             'south':'%.2f' % np.floor(ll.lat),
             'east':'%.2f' % np.ceil(ur.lon),
             'addLatLon':'true',
             'temporal':'all',
            }
    full_query = urllib.unquote(urllib.urlencode(query))
    # here we have the query for the actual dataset
    url = "%s?%s" % (url, full_query)
    logging.warn(url)

    encoded_name = "%s.nc" % str(uuid.UUID(bytes=hashlib.md5(url).digest()))
    dir = dir or 'ncdf_subset'
    path = path or os.path.join(_data_dir, dir, encoded_name)
    if not os.path.exists(path):
        logging.info("dumping forecast to: %s" % path)
        urlf = urllib2.urlopen(url)
        with open(path, 'wb') as f:
            f.write(urlf.read())

    logging.warn(path)
    obj = objects.Data(ncdf=open(path, 'r'))

    if 'ens' in obj.variables and obj['ens'].ndim != 1:
        obj.delete_variable('ens')

    return units.normalize_data(obj.renamed(vars))

def latest_nww3():
    index_url = 'http://motherlode.ucar.edu/thredds/catalog/fmrc/NCEP/WW3/Global/runs/catalog.html'
    data_url = 'http://motherlode.ucar.edu/thredds/ncss/grid/fmrc/NCEP/WW3/Global/runs/%s'
    print index_url
    f = urllib2.urlopen(index_url)
    soup = BeautifulSoup(f.read())
    text = [x.fetchText() for x in soup.findAll("a")]
    tags = [x[0].string for x in text if len(x)]
    return data_url % sorted(tags)[-1]

def latest_gefs():
    latest_url = 'http://motherlode.ucar.edu/thredds/catalog/NCEP/GEFS/Global_1p0deg_Ensemble/member/latest.html'
    dataset_base_url = 'http://motherlode.ucar.edu/thredds/ncss/grid/'
    print latest_url
    f = urllib2.urlopen(latest_url)
    soup = BeautifulSoup(f.read())
    def is_latest(x):
        # checks if the beautiful soup a tag holds the latest href
        text = x.fetchText()
        return len(text) == 1 and 'Latest' in str(text[0])
    atag = [x for x in soup.findAll("a") if is_latest(x)]
    if len(atag) != 1:
        raise ValueError("Expected at least one tag with Latest in the name:" +
                         "instead got %s" % str(latest_aref))
    atag = atag[0]
    (_, query) = urllib.splitquery(atag.get(u'href'))
    query = dict(urlparse.parse_qsl(urllib.splitquery(atag.get('href'))[1]))
    dataset = query['dataset']
    url = os.path.join(dataset_base_url, dataset)
    # this url should look like this
    #http://motherlode.ucar.edu/thredds/ncss/grid/NCEP/GEFS/Global_1p0deg_Ensemble/member/GEFS_Global_1p0deg_Ensemble_20110830_1800.grib2
    return url

def gefs_subset(ll, ur, url=None, path=None):
    """
    Global Ensemble Forecast System forecast object
    """

    vars = {'U-component_of_wind_height_above_ground':conv.UWND,
            'V-component_of_wind_height_above_ground':conv.VWND,}

    if not url and not path:
        url = latest_gefs()

    obj = ncdf_subset(url=url, ll=ll, ur=ur, vars=vars, dir='gefs', path=path)
    renames = {'ens':conv.ENSEMBLE}
    if 'time1' in obj.variables:
        renames['time1'] = conv.TIME
    obj = obj.renamed(renames)
    obj = obj.squeeze(dimension='height_above_ground1')
    gefs_format = 'hour since %Y-%m-%d %H:%M:%S'
    coards_format = 'hours since %Y-%m-%d %H:%M:%S'
    try:
        date = datetime.datetime.strptime(obj[conv.TIME].attributes[conv.UNITS],
                                          gefs_format)
    except:
        date = datetime.datetime.strptime(obj[conv.TIME].attributes[conv.UNITS],
                                          'hour since %Y-%m-%dT%H:%M:%SZ')
    obj[conv.TIME].attributes[conv.UNITS] = date.strftime(coards_format)

    # remove the ensemble variable since it contains unneeded text
    var = obj.create_variable(conv.ENSEMBLE, (conv.ENSEMBLE,),
                              data=np.arange(obj.dimensions[conv.ENSEMBLE]))
    obj.sort_coordinate(conv.LAT)
    obj.sort_coordinate(conv.LON)
    return obj

def nww3_subset(ll, ur, path=None):
    vars = {'Significant_height_of_combined_wind_waves_and_swell':'combined_swell_height',
            'U-component_of_wind':'uwnd',
            'V-component_of_wind':'vwnd',
            'Primary_wave_direction':'primary_wave_direction',
            'Direction_of_wind_waves':'direction_of_wind_waves'}
    url = 'http://motherlode.ucar.edu/thredds/ncss/grid/fmrc/NCEP/WW3/Global/runs/NCEP-WW3-Global_RUN_%Y-%m-%dT06:00:00Z'

    if not path:
        url = latest_nww3()
    return ncdf_subset(url=url, ll=ll, ur=ur, vars=vars, dir='nww3', path=path)

def ensure_corners(ur, ll, expand=True):
    """
    Makes sure the upper right (ur) corner is actually the upper right.  Same
    for the lower left (ll).
    """
    ur = ur.copy()
    ll = ll.copy()
    if ur.lon < ll.lon:
        tmp = ur.lon
        ur.lon = ll.lon
        ll.lon = tmp
    if ur.lat < ll.lat:
        tmp = ur.lat
        ur.lat = ll.lat
        ll.lat = tmp
    if expand:
        ur.lat += 1
        ur.lon += 1
        ll.lat -= 1
        ll.lon -= 1
    return ur, ll

def email_forecast(query, path=None):
    ll = objects.LatLon(query['lower'], query['left'])
    ur = objects.LatLon(query['upper'], query['right'])
    ur, ll = ensure_corners(ur, ll, expand=False)
    obj = gefs_subset(ll, ur, path=path)

    # the hours query is a bit complex since it
    # involves interpreting the '..' as slices
    all_inds = np.arange(obj.dimensions['time'])
    def time_ind_generator(hours_str):
        """Parses a saildocs string of desired hours"""
        for hr in hours_str.split(','):
            if '..' in hr:
                low, high = map(float, hr.split('..'))
                low_ind = np.nonzero(obj['time'].data == low)[0][0]
                high_ind = np.nonzero(obj['time'].data == high)[0][0]
                for x in all_inds[low_ind:(high_ind + 1)]:
                    yield x
            else:
                yield np.nonzero(obj['time'].data == float(hr))[0][0]
    time_inds = list(time_ind_generator(query['hours']))
    obj = obj.take(time_inds, 'time')

    lats = np.linspace(ll.lat, ur.lat,
                       num = (ur.lat - ll.lat) / query['grid_delta'] + 1,
                       endpoint=True)
    lons = np.linspace(ll.lon, ur.lon,
                       num = (ur.lon - ll.lon) / query['grid_delta'] + 1,
                       endpoint=True)
    lat_inds = np.concatenate([np.nonzero(obj['lat'].data == x)[0] for x in lats])
    if not lat_inds.size:
        raise ValueError("forecast lats don't overlap with the requested lats")
    lon_inds = np.concatenate([np.nonzero(obj['lon'].data == x)[0] for x in lons])
    if not lon_inds.size:
        raise ValueError("forecast lons don't overlap with the requested lons")
    obj = obj.take(lat_inds, 'lat')
    obj = obj.take(lon_inds, 'lon')
    return obj

def forecast_weather(start_date, ur, ll, recent=False):
    ur, ll = ensure_corners(ur, ll)

    def most_recent(dir):
        files = os.listdir(dir)
        if not len(files):
            raise ValueError("No files found in %s" % dir)
        def ctime(x):
            return os.path.getctime(os.path.join(dir, x))
        sfiles = sorted(files, key=ctime)
        return os.path.join(dir, sfiles[-1])

    paths = {'gefs':None, 'nww3':None}

    if recent:
        dirs = dict((x, os.path.join(_data_dir, x)) for x in ['gefs', 'nww3'])
        paths = dict((k, most_recent(d)) for k, d in dirs.iteritems())
        print paths

    gefs = gefs_subset(ur=ur, ll=ll, path=paths['gefs'])
    nww3 = nww3_subset(ur=ur, ll=ll, path=paths['nww3'])

    def daterange(obj):
        st = units.from_udunits(min(obj[conv.TIME]),
                                obj[conv.TIME].attributes[conv.UNITS])
        en = units.from_udunits(max(obj[conv.TIME]),
                                obj[conv.TIME].attributes[conv.UNITS])
        return [st, en]
    print "nww3 spans %s to %s" % tuple(str(x) for x in daterange(nww3))
    print "gefs spans %s to %s" % tuple(str(x) for x in daterange(gefs))
    gefs.sort_coordinate(conv.LAT)
    gefs.sort_coordinate(conv.LON)
    for ens, gfs in gefs.iterator(conv.ENSEMBLE):
        yield [gfs, nww3]

def historical_weather(start_date, data_dir=None, source='ccmp_daily'):
    if not data_dir:
        data_dir = _data_dir
    fn_format = os.path.join(data_dir, _sources[source])
    storms_path = os.path.join(data_dir, _storms)
    if not os.path.exists(storms_path):
        msg = "Missing data file: %s, Have you created the storms file yet?"
        raise IOError(msg % storms_path)
    storms_nc = objects.DataObject(filename=storms_path, mode='r')

    def historical_year(year):
        now = start_date.replace(year=year)
        for day in range(365):
            def get_data():
                filename = now.strftime(fn_format)
                if not os.path.exists(filename):
                    msg = "Missing data file: %s, Have you downloaded the data for %s yet?"
                    raise IOError(msg % (filename, source))
                nc = objects.DataObject(filename=filename, mode='r')
                nc = nc.slice('time', 0)

                timevar = nc.variables['time']
                # ccmp_daily is stored in files with one day per file
                assert timevar.shape[0] == 1
                # unfortunately the cdo operation invalidates the coard stored times
                timevar = datetime.datetime.strptime(str(timevar[0]), '%Y%m%d.%f')
                # make sure it the requested day
                assert timevar.strftime('%Y%m%d') == now.strftime('%Y%m%d')

                variables = ['uwnd', 'vwnd']
                yield dict((v, objects.DataField(nc, v)) for v in ['uwnd', 'vwnd'])
            yield now, get_data()
            now = now + datetime.timedelta(days=1)

    return [list(historical_year(year)) for year in range(2000, 2009)]

def main():

    ll = objects.LatLon(30., -115.)
    ur = objects.LatLon(40., -125.)
    obj = gefs_subset(ll, ur)
    from wx.lib import tinylib

    string = tinylib.tiny(obj, ['uwnd', 'vwnd'])
    new_obj = tinylib.huge(string)

    import zlib
    print len(zlib.compress(obj.dumps()))/ float(len(zlib.compress(string)))
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
