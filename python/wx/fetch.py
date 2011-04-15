import os
import numpy as np
import itertools

from ftplib import FTP

from wx.lib import pupynere, objects

_nasa_ftp = 'podaac.jpl.nasa.gov'
_level30_url = 'ocean_wind/ccmp/L3.0/data/flk/'

def fetch_ccmp():
    ftp = FTP(_nasa_ftp)
    ftp.login('anonymous', '')

    def list_ftp(dirname):
        dirs = []
        ftp.dir(dirname, dirs.append)
        dirs = [x.split(' ')[-1] for x in dirs]
        return [x for x in dirs if not x == 'README.txt']

    def decend_dir_structure():
        years = list_ftp(_level30_url)
        for year in years:
            url = os.path.join(_level30_url, year)
            months = list_ftp(url)
            for month in months:
                month_url = os.path.join(url, month)
                files = list_ftp(month_url)
                for f in files:
                    yield os.path.join('ftp://', _nasa_ftp,month_url, f)

    for f in decend_dir_structure():
        print f
    return 0

def storm_history_mapper():
    """
    Takes the IBTrACS tropical storm data set from:

    ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r02/wmo/netcdf/Allstorms.ibtracs_wmo.v03r02.nc.gz

    and maps each storms entry to a tuple indicating the day of the year
    and the lat long and wind speed of the storm on that day.
    Duplicate days of the year and locations for the same storm are skipped.
    """
    nc = pupynere.netcdf_file(os.path.join(_data_dir, _ibtracs), 'r')

    lats = nc.variables['lat_wmo']
    lats = lats.data * lats.scale_factor
    lons = nc.variables['lon_wmo']
    lons = lons.data * lons.scale_factor
    wind = nc.variables['wind_wmo']
    wind = wind.data * wind.scale_factor
    time = nc.variables['time_wmo']
    nobs = nc.variables['numObs']

    nstorms = nobs.shape[0]
    for storm in range(nstorms):
        prev = None
        def storm_track():
            for obs in range(nobs[storm]):
                doy = min(objects.from_udunits(time[storm, obs], time.units).timetuple().tm_yday, 365)
                yield doy, int(lats[storm, obs]), int(lons[storm, obs]), int(wind[storm, obs])

        for k, values in itertools.groupby(storm_track(), key=operator.itemgetter(0)):
            values = list(values)
            mean_lat = int(np.mean([x[1] for x in values]))
            mean_lon = int(np.mean([x[2] for x in values]))
            max_wind = int(max([x[3] for x in values]))
            yield k, mean_lat, mean_lon, max_wind

def create_storm_history():
    history = sorted(storm_history_mapper(), key=operator.itemgetter(0))

    output = os.path.join(_data_dir, 'storm_counts.nc')
    nc = pupynere.netcdf_file(output, 'w')

    nc.created = datetime.datetime.utcnow().isoformat()
    nc.createDimension("time", 365)
    nc.createDimension("latitude", 180)
    nc.createDimension("longitude", 360)

    m = nc.createVariable("time", "i", ("time",))
    m.long_name = "day of the year"
    m.units = "days since jan 1st"
    m[:] = np.arange(365) + 1

    m = nc.createVariable("latitude", "i", ("latitude",))
    m.long_name = "Latitude"
    m.units = "degrees_north"
    m.axis = "Y"
    m[:] = np.arange(-90, 90)

    m = nc.createVariable("longitude", "i", ("longitude",))
    m.long_name = "Longitude"
    m.units = "degrees_east"
    m.axis = "X"
    m[:] = np.arange(0, 360)

    counts = nc.createVariable('counts', "i", ("time", "latitude", "longitude"))
    winds = nc.createVariable('max_winds', "i", ("time", "latitude", "longitude"))

    for k, values in itertools.groupby(history, key=operator.itemgetter(0)):
        print k
        for doy, lat, lon, wind in values:
            for i in range(-5, 6):
                for j in range(-5, 6):
                    la = np.mod(lat + 90 + i, 180)
                    lo = np.mod(lon + j, 360)
                    counts[k - 1, la, lo] += 1
                    scale = (1. - np.sqrt(i*i + j*j) / (np.sqrt(2) * 5))
                    winds[k - 1, la, lo] = max(winds[k - 1, la, lo], wind*scale)

if __name__ == "__main__":
    import sys
    sys.exit(create_storm_history())

