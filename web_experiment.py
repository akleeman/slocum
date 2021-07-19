import os
import re
import json
import zlib
import shutil
import struct
import urllib
import cProfile
import requests
import warnings

import numpy as np
import xarray as xra
import multiprocessing as mp

from itertools import chain
from datetime import datetime, timedelta
from slocum.lib.units import convert_units
from slocum.compression import tinylib
from pathlib import Path

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

GEFS_MEMBERS = 31
FCST_RELEASE_TIMES = list(range(0, 240, 3))
#FCST_RELEASE_TIMES = list(range(0, 9, 3))


def list_of_tuples(xs):
    if isinstance(xs[0], tuple):
        return xs
    return [(x,) for x in xs]


def serial_apply(f, arguments):
    return [f(*x) for x in list_of_tuples(arguments)]


def parallel_apply(f, arguments):
    pool = mp.Pool(4)

    results = []
    def collect_result(result):
        if result:
            results.append(result)

    [pool.apply_async(f, args=x, callback=collect_result)
     for x in list_of_tuples(arguments)]
    pool.close()
    pool.join()
    return sorted(results)


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
    else:
        print("Cached: ", output_path)
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


def shrink(ds):
    ds = ds.sel(latitude=slice(65, -65))
    ds['speed'] = to_beaufort(ds['speed'])    
    ds['direction'] = to_integer_direction(ds['direction'])
    return ds.expand_dims('step')


def download_members(ref_time, fcst_hour, directory=None):

    if directory is None:
        directory = ref_time.strftime('./%Y%d%m_%H')
        if not os.path.exists(directory):
            os.mkdir(directory)

    combined_path = '%s_%.3d.nc' % (ref_time.strftime('%Y%d%m_%H'), fcst_hour)
    combined_path = os.path.join(directory, combined_path)
    if not os.path.exists(combined_path):
        paths = [download_one_file(member, ref_time, fcst_hour, directory)
                 for member in range(GEFS_MEMBERS)]
        ds = xra.concat([normalize(xra.open_dataset(p, engine='cfgrib'))
                         for p in paths],
                        dim='member')
        ds = shrink(ds)
        ds.to_netcdf(combined_path, encoding={'speed': {'zlib': True},
                                              'direction': {'zlib': True}})
        
    return combined_path


def download(ref_time, directory=None):

    return [download_members(ref_time, fcst_hour, directory)
            for fcst_hour in FCST_RELEASE_TIMES]


def open_files(nc_paths):
    return xra.open_mfdataset(nc_paths,
                              chunks={'latitude': 10, 'longitude': 10},
                              concat_dim='step', combine='nested')


def nomads_has_complete_forecast(time):
    _, url = build_url(1, time, max(FCST_RELEASE_TIMES))
    return requests.get(url).status_code == 200
    

def nearest_fcst_release(time):
    print("Current Time ", time)
    six_hours = timedelta(minutes=6*60)
    one_day = 4 * six_hours
    ref_time = time + six_hours
    ref_time = datetime(time.year, time.month, time.day, time.hour - time.hour % 6)
    while ref_time + one_day > time:
        print("Trying Time  ", ref_time)
        if nomads_has_complete_forecast(ref_time):
            return ref_time
        ref_time = ref_time - six_hours
    return None



def make_compressed(speeds, directions, path):
    first = True
    assert(speeds.shape == directions.shape)
    
    bytes = b''
    lats = speeds.coords['latitude'].values
    lons = speeds.coords['longitude'].values
    assert(speeds.dims == ('member', 'latitude', 'longitude'))
    speeds = speeds.values
    directions = directions.values
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            speed_vals = speeds[:, i, j]
            direction_vals = directions[:, i, j]
            packed = struct.pack("f", lon)
            packed += struct.pack("f", lat)
            packed += speed_vals.astype('uint8').tobytes()
            packed += direction_vals.astype('uint8').tobytes()

            if first:
              bytes += struct.pack("I", len(packed))
              first = False
            
            bytes += packed

    with open(path, 'wb') as f:
        f.write(bytes)



def make_spot(speeds, directions, hours, path):
    with open(path, 'wb') as f:
        encoded_time = np.concatenate([[hours.size], np.diff(hours)])
        packed = encoded_time.astype('uint8').tobytes()
        packed += speeds.astype('uint8').tobytes()
        packed += directions.astype('uint8').tobytes()        
        f.write(packed)



def make_spots(ds):
    speeds = ds['speed']
    directions = ds['direction']
    hours = ds['step'].values.astype('timedelta64[h]')
        
    lats = speeds.coords['latitude'].values
    lons = speeds.coords['longitude'].values
    assert(speeds.dims == ('step', 'member', 'latitude', 'longitude'))
    assert(directions.dims == speeds.dims)    
    speeds = speeds.values
    directions = directions.values
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            make_spot(speeds[:, :, i, j],
                      directions[:, :, i, j],
                      hours, './data/spot/%.3f_%.3f.bin' % (lon, lat))


def make_slices(x, k):

    chunks = np.array_split(x, k)
    
    i = 0
    for chunk in chunks:
        yield slice(i, i + chunk.size)
        i += chunk.size


def get_chunk_dir(i, j):
    return './chunks/{}_{}'.format(i, j)


def split_one_dataset(path, lat_slices, lon_slices):
    print("Splitting : ", path)
    ds = xra.load_dataset(path)
    for i, lat_slice in enumerate(lat_slices):
        for j, lon_slice in enumerate(lon_slices):
            fcst_hour = ds['step'].values.item()
            output_path = os.path.join(get_chunk_dir(i, j), '{}.nc'.format(fcst_hour))
            ds.isel(latitude=lat_slice, longitude=lon_slice).to_netcdf(output_path)


def split_into_chunks(paths):
    
    peek = xra.open_dataset(paths[0])
    lat_slices = list(make_slices(peek['latitude'].values, 10))
    lon_slices = list(make_slices(peek['longitude'].values, 10))    

    def make_chunk_directories():
        for i in range(len(lat_slices)):
            for j in range(len(lon_slices)):
                directory = get_chunk_dir(i, j)
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                Path(directory).mkdir(parents=True, exist_ok=False)
                yield directory

    output = list(make_chunk_directories())

    arguments = [(p, lat_slices, lon_slices) for p in paths]

    parallel_apply(split_one_dataset, arguments)

    return output


def make_spots_one_chunk(chunk_dir):
    print("Making spots : ", chunk_dir)
    ds = xra.open_mfdataset(os.path.join(chunk_dir, '*.nc'),
                            concat_dim='step')
    ds.load()
    make_spots(ds)


def make_all_spots(paths):
    chunk_directories = split_into_chunks(paths)
    arguments = [(x,) for x in chunk_directories]
    parallel_apply(make_spots_one_chunk, arguments)
    

def make_zoom_level_one_step(ds, directory, zoom, spacing):
    number_of_tiles = 2 ** (zoom - 1)

    ds = ds.isel(latitude=slice(0, ds.dims['latitude'], spacing),
                 longitude=slice(0, ds.dims['longitude'], spacing))

    max_latitude = 85

    mercator_max_y = np.log(np.tan(np.pi / 4. + np.deg2rad(max_latitude) / 2))    
    mercator_y_bins = np.linspace(mercator_max_y, -mercator_max_y, number_of_tiles + 1)    
    lat_bins = np.rad2deg(2 * np.arctan(np.exp(mercator_y_bins)) - np.pi / 2)
    lon_bins = np.linspace(-180, 180, number_of_tiles + 1)
    
    shifted_lons = np.mod(ds['longitude'].values + 180., 360.) - 180.
    
    ds['lat_bin'] = ('latitude', np.digitize(ds['latitude'].values, lat_bins, right=True) - 1)
    ds['lon_bin'] = ('longitude', np.digitize(shifted_lons, lon_bins) - 1)

    assert(ds['lon_bin'].values.min() >= 0)
    assert(ds['lon_bin'].values.max() < number_of_tiles)
    assert(ds['lat_bin'].values.min() >= 0)
    assert(ds['lat_bin'].values.max() < number_of_tiles)

    fcst_hour = ds['step'].values.astype('timedelta64[h]').astype('int')
    sub_dir = os.path.join(directory, str(zoom), str(fcst_hour))
    Path(sub_dir).mkdir(parents=True, exist_ok=True)    

    for lon_bin, lon_df in ds.groupby('lon_bin'):
        for lat_bin, tile_df in lon_df.groupby('lat_bin'):
            filename = '%d_%d.bin' % (lon_bin, lat_bin)
            path = os.path.join(sub_dir, filename)
            make_compressed(tile_df["speed"], tile_df["direction"], path)


def download_and_make_zoom_levels_one_step(ref_time, fcst_hour, output_directory, download_directory):
    path = download_members(ref_time, fcst_hour, download_directory)

    zoom_levels = {
      4: 16,
      5: 8,
      6: 4,
      7: 2,
      8: 1,
    }

    print("Forecast hour: ", fcst_hour)
    #[make_zoom_level_one_step(ds, output_directory, zoom, spacing)
    # for zoom, spacing in zoom_levels.items()]

    return path


def make_zoom_levels(ref_time, output_directory, download_directory=None, parallel=True):

    arguments = [(ref_time, fcst_hour, output_directory, download_directory)
                 for fcst_hour in FCST_RELEASE_TIMES]

    apply_func = parallel_apply if parallel else serial_apply
    return apply_func(download_and_make_zoom_levels_one_step, arguments)


def write_forecast_hours(ref_time, directory):
    with open(os.path.join(directory, 'time.bin'), 'wb') as f:
        f.write(struct.pack('I', int(ref_time.timestamp())))
        f.write(struct.pack('I', len(FCST_RELEASE_TIMES)))
        for dh in np.diff(FCST_RELEASE_TIMES):
            f.write(struct.pack('B', dh))


def main():

    # https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?dir=%2Fgefs.20210622%2F18
    # ref_time = nearest_fcst_release(datetime.utcnow())
    ref_time = datetime(2021, 7, 18, 6)

    directory = './data'
    write_forecast_hours(ref_time, directory)    

    step_paths = make_zoom_levels(ref_time, directory)    

    make_all_spots(step_paths)


if __name__ == "__main__":
    main()
