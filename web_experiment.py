import os
import json
import shutil
import warnings
import urllib.request

import numpy as np
import xarray as xra

from itertools import chain
from datetime import datetime, timedelta
from slocum.lib.units import convert_units
from slocum.compression import tinylib

# Can download the grib files from here:
# https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl

# Wind uses the beaufort scale for compression
wind_bins = np.array([0., 1., 3., 6., 10., 16., 21., 27.,
                      33., 40., 47., 55., 63., 75.])
wind_bin_names = ['F-{0:<12}'.format(i) for i in range(wind_bins.size - 1)]

dir_bins = np.linspace(-15 * np.pi/16., 15 * np.pi/16., 16)

velocity_colors = [
    '#d7d7d7',  # light grey
    '#a1eeff',  # lightest blue
    '#42b1e5',  # light blue
    '#4277e5',  # pastel blue
    '#60fd4b',  # green
    '#1cea00',  # yellow-green
    '#fbef36',  # yellow
    '#fbc136',  # orange
    '#ff4f02',  # red
    '#d50c02',  # darker-red
    '#ff00c0',  # red-purple
    '#b30d8a',  # dark purple
    '#000000',  # black
]

FCST_RELEASE_TIMES = [0, 6, 12, 18, 24, 30, 36, 42, 48,
                      54, 60, 66, 72, 78, 84, 90, 96]


def radial_point(center, radius, angle):
    return center + radius * np.array([np.sin(angle), np.cos(angle)])


def shapely_wind_slice(center, radius, speed, theta, width=np.pi / 16):
    color = 'blue'  # get_color(speed)
    poly = Polygon([center,
                    radial_point(center, radius, theta - width),
                    radial_point(center, radius, theta),
                    radial_point(center, radius, theta + width)])
    return poly


def wind_slice(lat, lon, radius, speed, theta, width=np.pi / 16):
    x_0, y_0 = lon, lat
    y_1, x_1 = radial_point([lat, lon], radius, theta - width)
    y_2, x_2 = radial_point([lat, lon], radius, theta)
    y_3, x_3 = radial_point([lat, lon], radius, theta + width)

    color = velocity_colors[np.digitize(speed, wind_bins)]

    return """var polygon = L.polygon([
        [{x_0}, {y_0}],
        [{x_1}, {y_1}],
        [{x_2}, {y_2}],
        [{x_3}, {y_3}]
    ], {{color: '{color}', fillOpacity:0.3, stroke:false}}).addTo(map); """.format(x_0=x_0, y_0=y_0,
                                                                                   x_1=x_1, y_1=y_1,
                                                                                   x_2=x_2, y_2=y_2,
                                                                                   x_3=x_3, y_3=y_3,
                                                                                   color=color)


def circle(lat, lon, radius):

    radius = radius * 60 * 1852

    return """var circle = L.circle([{x}, {y}], {radius}, {{
        stroke: false,
        fillColor: '#FFFFFF',
        fillOpacity: 0.9
    }}).addTo(map);""".format(x=lon, y=lat, radius=radius)


def leaflet_wind_circle(x, y, speeds, directions, radius):
    center = np.array([x, y]).flatten()
    radius = radius
    speeds = speeds.reshape(-1)
    inds = np.argsort(speeds)
    speeds = speeds[inds]
    directions = directions.reshape(-1)[inds]

    wind_slices = [wind_slice(*center, radius, ws, wd)
                   for ws, wd in zip(speeds, directions)]

    background_circle = circle(*center, radius)

    return '\n\n'.join(chain([background_circle], wind_slices))


def vector_to_radial(u, v, orientation='to'):
    """
    Converts from a vector variable with zonal (u) and
    meridianal (v) components to magnitude and direction
    from north.
    """
    assert orientation in ['from', 'to']
    # convert to magnitudes
    magnitude = np.sqrt(np.power(u, 2) + np.power(v, 2))
    direction = np.arctan2(u, v)
    if orientation == 'from':
        # note this is like: direction + pi but with modulo between
        direction = np.mod(direction + 2 * np.pi, 2 * np.pi) - np.pi
    return magnitude, direction


def build_url(member, ref_time, fcst_hour, left_lon=-160, right_lon=-140,
              top_lat=25, bottom_lat=15):
    member = str(member)
    if member.isdigit():
        if member == '0':
            member = 'c00'
        else:
            member = 'p%.2d' % int(member)

    fcst_hour = '%.3d' % fcst_hour

    date = ref_time.strftime('%Y%m%d')
    hour = ref_time.strftime('%H')

    file_name = 'ge{member}.t{hour}z.pgrb2s.0p25.f{fcst_hour}'.format(
        member=member,
        hour=hour,
        fcst_hour=fcst_hour)

    return (file_name, 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?file={file_name}&lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on&leftlon={left_lon}&rightlon={right_lon}&toplat={top_lat}&bottomlat={bottom_lat}&dir=%2Fgefs.{date}%2F{hour}%2Fatmos%2Fpgrb2sp25'.format(
        file_name=file_name,
        hour=hour,
        fcst_hour=fcst_hour,
        date=date,
        left_lon=left_lon,
        right_lon=right_lon,
        top_lat=top_lat,
        bottom_lat=bottom_lat
    ))


def download_one_file(member, ref_time, fcst_hour, directory):
    file_name, url = build_url(member, ref_time, fcst_hour)

    output_path = os.path.join(directory, file_name)
    if not os.path.exists(output_path):
        try:
            with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
                print("Downloaded: ", output_path)
        except:
            print("Failure Downloading: ", url)
            raise
    return output_path


def to_beaufort(knots):
    beaufort = np.digitize(knots.values.reshape(-1),
                           wind_bins).reshape(knots.shape)
    attrs = knots.attrs
    attrs['units'] = 'beaufort'
    return (knots.dims, beaufort.astype('u8'), attrs)


def from_beaufort(beaufort):
    divs = wind_bins
    ndivs = divs.size
    bins = beaufort.values
    upper_bins = bins
    lower_bins = np.maximum(upper_bins - 1, 0)
    upper_bins = np.minimum(upper_bins, ndivs - 1)
    averages = 0.5 * (divs[lower_bins] + divs[upper_bins])
    # if any values fell outside the divs we set them to be
    # slightly larger than div boundaries. Ideally to avoid
    # this the divs should be designed such that no values
    # fall beyond their range.
    epsilon = np.maximum(0.5 * np.min(np.diff(divs)), 1e-6)
    if np.any(bins == 0):
        averages[bins == 0] = np.min(divs) - epsilon
        warnings.warn("Some values were too small to decode!")
    if np.any(bins == ndivs):
        averages[bins == ndivs] = np.max(divs) + epsilon
        warnings.warn("Some values were too large to decode!")
    attrs = beaufort.attrs
    attrs['units'] = 'knots'
    return (beaufort.dims, averages, attrs)


def from_integer_direction(direction):
    divs = dir_bins
    bins = direction.values
    upper = bins % len(divs)
    wrap_val = -np.pi
    lower = (bins-1) % len(divs)
    # any bins that are zero mean they fell on the wrap value
    # all others are as before
    averages = np.where(bins > 0,
                        0.5 * (divs[lower] + divs[upper]),
                        wrap_val)
    attrs = direction.attrs
    attrs['units'] = 'radians'
    return (direction.dims, averages, attrs)


def to_integer_direction(direction):
    ints = np.digitize(direction.values.reshape(-1), dir_bins) % len(dir_bins)
    ints = ints.reshape(direction.shape)
    return (direction.dims, ints, direction.attrs)


def normalize(x):
    speed, direction = vector_to_radial(x['u10'], x['v10'], orientation='from')
    speed.attrs['units'] = 'm s**-1'
    direction.attrs.clear()
    x = x.drop_vars(['u10', 'v10'])

    x['speed'] = convert_units(speed, 'knots')
    #x['speed'] = to_beaufort(x['speed'])

    x['direction'] = direction
    #integer_direction = np.digitize(direction.values.reshape(-1), dir_bins).reshape(direction.shape)
    #x['direction'] = (direction.dims, integer_direction.astype('u8'), {'units': 'direction'})

    return x


def download_members(ref_time, fcst_hour, directory='./'):

    combined_path = '%s_%.3d.nc' % (ref_time.strftime('%Y%d%m_%H'), fcst_hour)
    combined_path = os.path.join(directory, combined_path)
    if not os.path.exists(combined_path):
        paths = [download_one_file(member, ref_time, fcst_hour, directory)
                 for member in range(31)]
        ds = xra.concat([normalize(xra.open_dataset(p, engine='cfgrib'))
                         for p in paths],
                        dim='member')
        ds.to_netcdf(combined_path, encoding={'speed': {'zlib': True},
                                              'direction': {'zlib': True}})

    return combined_path


def download(ref_time, directory=None):

    if directory is None:
        directory = ref_time.strftime('./%Y%d%m_%H')
        if not os.path.exists(directory):
            os.mkdir(directory)

    combined_path = os.path.join(directory, ref_time.strftime('%Y%d%m_%H.nc'))
    if not os.path.exists(combined_path):
        nc_paths = [download_members(ref_time, fcst_hour, directory)
                    for fcst_hour in FCST_RELEASE_TIMES]
        ds = xra.concat([xra.open_dataset(p) for p in nc_paths], dim='step')
        ds.to_netcdf(combined_path, encoding={'speed': {'zlib': True},
                                              'direction': {'zlib': True}})
    return combined_path
    

def nomads_url_exists(time):
    date = ref_time.strftime('%Y%m%d_%H')
    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?dir=gefs.{date}".format(
      date
    )
    

def nearest_fcst_release(time):
    six_hours = timedelta(minutes=6*60)
    time = time + six_hours
    ref_time = datetime(time.year, time.month, time.day, time.hour)
    while ref_time + six_hours < time:
        ref_time = ref_time + six_hours
    return ref_time


def iter_wind_circles(ds, radius=0.05):
    for lat in ds['latitude']:
        for lon in ds['longitude']:
            point = ds.sel(latitude=lat, longitude=lon)

            x = lon.values.item()
            y = lat.values.item()
            speeds = point['speed'].values
            directions = point['direction'].values

            yield leaflet_wind_circle(x, y, speeds, directions, radius)


def make_compressed(ds, path):
    first = True
    with open(path, 'wb') as f:
        for lat, lat_slice in ds.groupby('latitude'):
            for lon, point in lat_slice.groupby('longitude'):
                point = point.squeeze('latitude')
                speeds = point['speed'].values
                directions = point['direction'].values
                packed = np.array([lon, lat]).astype('float32').tobytes()
                packed += speeds.astype('uint8').tobytes()
                packed += directions.astype('uint8').tobytes()

                if first:
                  f.write(np.array(len(packed), dtype='uint32').tobytes())
                  first = False
                
                f.write(packed)



def make_spot(ds, path):
    with open(path, 'wb') as f:
        hours = ds['step'].values.astype('timedelta64[h]')
        speeds = ds['speed'].values
        directions = ds['direction'].values
        encoded_time = np.concatenate([[hours.size], np.diff(hours)])
        packed = encoded_time.astype('uint8').tobytes()
        packed += speeds.astype('uint8').tobytes()
        packed += directions.astype('uint8').tobytes()        
        f.write(packed)
        

def main():

    # https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?dir=%2Fgefs.20210622%2F18
    ref_time = nearest_fcst_release(datetime.utcnow())

    ref_time = datetime(2021, 6, 27, 18)

    path = download(ref_time)
    ds = xra.open_dataset(path)

    ds = ds.sel(latitude=slice(45, 15), longitude=slice(180, 300))
    ds['speed'] = to_beaufort(ds['speed'])    
    ds['direction'] = to_integer_direction(ds['direction'])

    if not os.path.exists('./data'):
        os.mkdir('./data')

    for lat, lat_slice in ds.groupby('latitude'):
        for lon, point in lat_slice.groupby('longitude'):
            make_spot(point.squeeze('latitude'), './data/%.3f_%.3f.bin' % (lon, lat))

    for step in ds['step'].values:
        fcst_hour = step.astype('timedelta64[h]').astype('int')
        make_compressed(ds.sel(step=step), './data/%.3d.bin' % fcst_hour)
        # make_leaflet(ds.sel(step=step), '%.3d.js' % fcst_hour)


if __name__ == "__main__":
    main()