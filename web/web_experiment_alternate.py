import os
import re
import sys
import json
import zlib
import shutil
import struct
import urllib
import cProfile
import requests
import warnings
import argparse

import numpy as np
import xarray as xra
import multiprocessing as mp

from io import BytesIO
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
FCST_RELEASE_TIMES = list(range(0, 170, 3))

def list_of_tuples(xs):
    if isinstance(xs[0], tuple):
        return xs
    return [(x,) for x in xs]


def serial_apply(f, arguments):
    return [f(*x) for x in list_of_tuples(arguments)]


def parallel_apply(f, arguments, n=2):
    pool = mp.Pool(n)

    results = []
    def collect_result(result):
        if result:
            results.append(result)

    [pool.apply_async(f, args=x, callback=collect_result)
     for x in list_of_tuples(arguments)]
    pool.close()
    pool.join()
    return sorted(results)


def maybe_parallel_apply(f, arguments, n=1):
    if n > 1:
        return parallel_apply(f, arguments, n)
    else:
        return serial_apply(f, arguments)


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


def build_url(member, ref_time, fcst_hour):
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


    return (file_name, 'https://ftp.ncep.noaa.gov/data/nccf/com/gens/prod/gefs.{date}/{hour}/atmos/pgrb2sp25/{file_name}'.format(
        file_name=file_name,
        hour=hour,
        date=date,
    ))


def read_wind_grib(path):
    # The grib file holds a LOT of variables, here we only want to read in
    # wind, the `filter_by_keys` arguments do this in a kinda round about way.
    return xra.open_dataset(path,
                            engine='cfgrib',
                            backend_kwargs={'filter_by_keys':
                                           {'typeOfLevel': 'heightAboveGround',
                                            'level': 10}})


def wind_grib_file_exists(path):
    if not os.path.exists(path):
        return False

    try:
        read_wind_grib(path)
    except:
        return False
    else:
        return True    


def download_one_file(member, ref_time, fcst_hour, directory):
    file_name, url = build_url(member, ref_time, fcst_hour)

    output_path = os.path.join(directory, file_name)

    if wind_grib_file_exists(output_path):
        print("Cached: ", output_path)
    else:
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
    x['direction'] = direction

    return x


def shrink(ds):
    ds = ds.sel(latitude=slice(80, -80))
    ds['speed'] = to_beaufort(ds['speed'])    
    ds['direction'] = to_integer_direction(ds['direction'])
    return ds.expand_dims('step')
    

def download_members(ref_time, fcst_hour, output_path, n_cpu, keep_cache):

    if os.path.exists(output_path):
        print("Using existing output: ", output_path)
    else:
        download_directory = os.path.dirname(output_path)
        
        args = [(member, ref_time, fcst_hour, download_directory)
               for member in range(GEFS_MEMBERS)]
        paths = maybe_parallel_apply(download_one_file, args)
        
        ds = xra.concat([normalize(read_wind_grib(p))
                         for p in paths],
                        dim='member')
        ds = shrink(ds)
        ds.to_netcdf(output_path, encoding={'speed': {'zlib': True},
                                            'direction': {'zlib': True}})
        
        if not keep_cache:
            [os.remove(p) for p in paths if os.path.exists(p)]
            index_paths = ['%s.9810b.idx' % p for p in paths]
            [os.remove(p) for p in index_paths if os.path.exists(p)]

    return output_path


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

    return


def make_spot(speeds, directions, hours, path):
    with open(path, 'wb') as f:
        encoded_time = np.concatenate([[hours.size], np.diff(hours)])
        packed = encoded_time.astype('uint8').tobytes()
        packed += speeds.astype('uint8').tobytes()
        packed += directions.astype('uint8').tobytes()        
        f.write(packed)



def make_spots(ds, directory):
    speeds = ds['speed']
    directions = ds['direction']
    hours = ds['step'].values.astype('timedelta64[h]')
    Path(directory).mkdir(parents=True, exist_ok=True)
        
    lats = speeds.coords['latitude'].values
    lons = speeds.coords['longitude'].values
    assert(speeds.dims == ('step', 'member', 'latitude', 'longitude'))
    assert(directions.dims == speeds.dims)    
    speeds = speeds.values
    directions = directions.values
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            path = os.path.join(directory, '%.3f_%.3f.bin' % (lon, lat))
            make_spot(speeds[:, :, i, j],
                      directions[:, :, i, j],
                      hours, path)


def make_slices(x, k):

    chunks = np.array_split(x, k)
    
    i = 0
    for chunk in chunks:
        yield slice(i, i + chunk.size)
        i += chunk.size


def get_chunk_dir(i, j, root_dir="./"):
    return os.path.join(root_dir, 'chunks/{}_{}'.format(i, j))


def split_one_dataset(path, lat_slices, lon_slices, root_dir):
    print("Splitting : ", path)
    ds = xra.load_dataset(path)
    for i, lat_slice in enumerate(lat_slices):
        for j, lon_slice in enumerate(lon_slices):
            fcst_hour = ds['step'].values.item()
            output_path = os.path.join(get_chunk_dir(i, j, root_dir), '{}.nc'.format(fcst_hour))
            ds.isel(latitude=lat_slice, longitude=lon_slice).to_netcdf(output_path)


def split_into_chunks(paths, download_dir):
    
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

    arguments = [(p, lat_slices, lon_slices, download_dir) for p in paths]

    maybe_parallel_apply(split_one_dataset, arguments)

    return output


def make_spots_one_chunk(chunk_dir, data_dir):
    spot_dir = os.path.join(data_dir, 'spot')
    print("Making spots : ", chunk_dir, spot_dir)
    ds = xra.open_mfdataset(os.path.join(chunk_dir, '*.nc'))
    ds.load()
    make_spots(ds, spot_dir)


def make_all_spots(paths, data_dir, keep_cache, n_cpu, download_dir):
    chunk_directories = split_into_chunks(paths, download_dir)
    arguments = [(x, data_dir) for x in chunk_directories]

    serial_apply(make_spots_one_chunk, arguments)
    if not keep_cache:
        [shutil.rmtree(d) for d in chunk_directories]
    

def make_zoom_level_one_step(ds, directory, zoom, spacing):
    if 'step' in ds['speed'].dims:
        ds = ds.squeeze('step')
        
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
    assert(fcst_hour.size == 1)
    fcst_hour = fcst_hour.item()

    sub_dir = os.path.join(directory, str(zoom), str(fcst_hour))
    Path(sub_dir).mkdir(parents=True, exist_ok=True)

    for lon_bin, lon_df in ds.groupby('lon_bin'):
        for lat_bin, tile_df in lon_df.groupby('lat_bin'):
            filename = '%d_%d.bin' % (lon_bin, lat_bin)
            path = os.path.join(sub_dir, filename)
            make_compressed(tile_df["speed"], tile_df["direction"], path)


def download_and_make_zoom_levels_one_step(ref_time, fcst_hour, output_directory, combined_path, n_cpu, keep_cache):
    print("Downloading: ", fcst_hour)
    
    download_members(ref_time, fcst_hour, combined_path, n_cpu, keep_cache)

    zoom_levels = {
      4: 16,
      5: 8,
      6: 4,
      7: 2,
      8: 1,
    }

    print("Loading: ", fcst_hour)    
    ds = xra.load_dataset(combined_path)

    print("Making Tiles: ", fcst_hour)
    
    args = [(ds, output_directory, zoom, spacing) for zoom, spacing in zoom_levels.items()]
    maybe_parallel_apply(make_zoom_level_one_step, args, n_cpu)


def make_zoom_levels(ref_time, output_directory, download_directory=None, n_cpu=1, keep_cache=False):
    if download_directory is None:
        download_directory = "./"

    date_dir = os.path.join(download_directory, ref_time.strftime('%Y%m%d_%H'))
    Path(date_dir).mkdir(parents=True, exist_ok=True)

    def get_output_path(hour):
        path = '%s_%.3d.nc' % (ref_time.strftime('%Y%d%m_%H'), hour)
        return os.path.join(date_dir, path)

    arguments = [(ref_time, fcst_hour, output_directory, get_output_path(fcst_hour), n_cpu, keep_cache)
                 for fcst_hour in FCST_RELEASE_TIMES]

    serial_apply(download_and_make_zoom_levels_one_step, arguments)
    
    return [path for _, _, _, path, _ in arguments]


def write_forecast_hours(ref_time, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)    
    with open(os.path.join(directory, 'time.bin'), 'wb') as f:
        f.write(struct.pack('I', int(ref_time.timestamp())))
        f.write(struct.pack('I', len(FCST_RELEASE_TIMES)))
        for dh in np.diff(FCST_RELEASE_TIMES):
            f.write(struct.pack('B', dh))


def main(args):

    # https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?dir=%2Fgefs.20210622%2F18
    ref_time = nearest_fcst_release(datetime.utcnow())

    data_directory = args.output
    write_forecast_hours(ref_time, data_directory)    

    step_paths = make_zoom_levels(ref_time, data_directory,
                                  download_directory=args.download_dir,
                                  n_cpu=args.p,
                                  keep_cache=args.cache)    

    make_all_spots(step_paths, data_directory,
                   keep_cache=args.cache, n_cpu=args.p,
                   download_dir=args.download_dir)

    if not args.cache:
        [os.remove(p) for p in step_paths if os.path.exists(p)]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--p', default=1, type=int)
    p.add_argument('--cache', default=False, action="store_true")
    p.add_argument('--output', default="./data")
    p.add_argument('--download-dir', default="./")
    main(p.parse_args())
