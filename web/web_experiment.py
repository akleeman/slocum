"""
Web experiment for processing wind data from zarr sources.

This module uses the slocum conventions module to ensure consistent
variable naming and coordinate handling across different data sources.
"""

import os
import shutil
import warnings
import zarr
import pandas as pd

import numpy as np
import xarray as xra
import multiprocessing as mp

import itertools
from datetime import datetime, timedelta, timezone
from slocum.lib.units import convert_units
from slocum.compression import tinylib
from slocum.lib import conventions
from pathlib import Path
from dask.diagnostics import ProgressBar

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

# Zarr sources for ensemble forecasts
ZARR_SOURCES = {
    'dynamical': 'https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr',
    # Add more zarr sources as needed
}

# Default source to use
DEFAULT_SOURCE = 'dynamical'

def list_of_tuples(xs):
    if isinstance(xs[0], tuple):
        return xs
    return [(x,) for x in xs]


def serial_apply(f, arguments):
    return [f(*x) for x in list_of_tuples(arguments)]


def parallel_apply(f, arguments):
    pool = mp.Pool(2)

    results = []
    def collect_result(result):
        if result:
            results.append(result)

    [pool.apply_async(f, args=x, callback=collect_result)
     for x in list_of_tuples(arguments)]
    pool.close()
    pool.join()
    return sorted(results)


def convert_units_lazy(da, target_units):
    """
    Dask-compatible unit conversion that maintains laziness.
    
    Parameters:
    -----------
    da : xarray.DataArray
        DataArray with units to convert
    target_units : str
        Target units to convert to
        
    Returns:
    --------
    xarray.DataArray
        New DataArray with converted units (dask-backed if input was dask-backed)
    """
    if 'units' not in da.attrs:
        raise ValueError("No units found in DataArray attributes")
    
    current_units = da.attrs['units']
    
    # Find conversion factor from slocum units
    from slocum.lib.units import _all_units
    conversion_factor = None
    
    for (possible_units, _, _) in _all_units:
        if current_units in possible_units and target_units in possible_units:
            conversion_factor = possible_units[current_units] / possible_units[target_units]
            break
    
    if conversion_factor is None:
        raise ValueError(f"Cannot convert from {current_units} to {target_units}")
    
    # If conversion factor is 1, just update attributes
    if conversion_factor == 1.0:
        new_da = da.copy()
        new_da.attrs = da.attrs.copy()
        new_da.attrs['units'] = target_units
        return new_da
    
    # For dask arrays, use map_blocks to maintain laziness
    if hasattr(da.data, 'map_blocks'):
        def convert_block(block):
            return block * conversion_factor
        
        converted_data = da.data.map_blocks(
            convert_block,
            dtype=da.dtype,
            drop_axis=None
        )
    else:
        # For non-dask arrays, direct multiplication
        converted_data = da.data * conversion_factor
    
    # Create new DataArray with converted data
    new_attrs = da.attrs.copy()
    new_attrs['units'] = target_units
    
    return xra.DataArray(
        converted_data,
        dims=da.dims,
        coords=da.coords,
        attrs=new_attrs,
        name=da.name
    )


def vector_to_radial(u, v, orientation='to'):
    """
    Converts from a vector variable with zonal (u) and
    meridianal (v) components to magnitude and direction
    from north. Dask-compatible.
    """
    assert orientation in ['from', 'to']
    # convert to magnitudes - works with dask arrays
    magnitude = np.sqrt(u**2 + v**2)
    direction = np.arctan2(u, v)
    if orientation == 'from':
        # note this is like: direction + pi but with modulo between
        direction = np.mod(direction + 2 * np.pi, 2 * np.pi) - np.pi
    return magnitude, direction


def find_latest_zarr_source(source_name=DEFAULT_SOURCE):
    """
    Find the most recent zarr dataset from the specified source.
    For now, we'll use a known working zarr endpoint.
    """
    source_url = ZARR_SOURCES.get(source_name)
    if not source_url:
        raise ValueError(f"Unknown source: {source_name}")
    
    return source_url


def fetch_wind_data(source_name=DEFAULT_SOURCE, variables=None):
    """
    Fetch wind data from zarr source and convert to speed/direction.
    
    Parameters:
    -----------
    source_name : str
        Name of the zarr source to use
    variables : list
        List of variables to fetch. Defaults to wind components.
    
    Returns:
    --------
    xarray.Dataset
        Dataset containing wind speed and direction
    """
    if variables is None:
        # Use standard wind component names from conventions
        variables = [conventions.U_WIND, conventions.V_WIND]
    
    zarr_url = find_latest_zarr_source(source_name)
    print(f"Fetching from zarr source: {zarr_url}")
    
    # Open zarr with chunks preserved - this stays lazy
    in_ds = xra.open_zarr(zarr_url, decode_timedelta=True, chunks='auto')
    print(f"Opened zarr: {zarr_url}")
    
    in_ds = in_ds.isel(init_time=slice(0, 1))
    print(f"Selected init_time: {in_ds.init_time}")

    desired_variables = set(["wind_u_10m", "wind_v_10m"])

    for var in list(desired_variables):
        for d in in_ds[var].dims:
            desired_variables.add(d)

    in_ds = in_ds.drop_vars(list(in_ds.data_vars.keys() - desired_variables))

    for var in in_ds.data_vars:
        in_ds[var] = in_ds[var].astype('float16')

    return in_ds


def to_beaufort(knots):
    """Convert wind speeds to Beaufort scale (lazy version)."""
    import dask.array as da
    
    # Work with dask arrays to maintain laziness
    if hasattr(knots, 'data') and hasattr(knots.data, 'map_blocks'):
        # This is a dask-backed xarray
        def _digitize_block(block):
            return np.digitize(block, wind_bins).astype('uint8')
        
        beaufort_data = knots.data.map_blocks(
            _digitize_block, 
            dtype='uint8', 
            drop_axis=None
        )
    else:
        # Fallback to immediate computation if not dask-backed
        beaufort_data = np.digitize(knots.values, wind_bins).astype('uint8')
    
    attrs = knots.attrs.copy()
    attrs['units'] = 'beaufort'
    return (knots.dims, beaufort_data, attrs)


def from_beaufort(beaufort):
    """Convert Beaufort scale back to wind speeds."""
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
    """Convert integer direction bins back to radians."""
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
    """Convert wind directions to integer bins (lazy version)."""
    import dask.array as da
    
    # Work with dask arrays to maintain laziness
    if hasattr(direction, 'data') and hasattr(direction.data, 'map_blocks'):
        # This is a dask-backed xarray
        def _digitize_direction_block(block):
            ints = np.digitize(block, dir_bins) % len(dir_bins)
            return ints.astype('uint8')
        
        direction_data = direction.data.map_blocks(
            _digitize_direction_block,
            dtype='uint8',
            drop_axis=None
        )
    else:
        # Fallback to immediate computation if not dask-backed
        ints = np.digitize(direction.values, dir_bins) % len(dir_bins)
        direction_data = ints.astype('uint8')
    
    attrs = direction.attrs.copy()
    return (direction.dims, direction_data, attrs)


def compress_wind_data(ds):
    """
    Convert wind data using Beaufort scale and integer direction compression.
    """
    print("Before compression:")
    print(f"  Coordinates: {list(ds.coords.keys())}")
    print(f"  Data variables: {list(ds.data_vars.keys())}")
    print(f"  Dimensions: {dict(ds.dims)}")
    
    ds_compressed = ds.copy()
    
    # Get wind component names from the dataset
    u_name, v_name = conventions.get_wind_component_names(ds_compressed)
    
    if u_name and v_name:
        print(f"Found wind components: {u_name}, {v_name}")
        
        # Get wind components
        u_wind = ds_compressed[u_name]
        v_wind = ds_compressed[v_name]
        
        # Convert from m/s to knots if needed
        if u_wind.attrs.get('units', '') == 'm/s':
            print("Converting wind from m/s to knots")
            u_wind = convert_units_lazy(u_wind, 'knots')
            v_wind = convert_units_lazy(v_wind, 'knots')
        
        # Convert to speed and direction
        print("Converting to speed and direction...")
        speed, direction = vector_to_radial(u_wind, v_wind, orientation='to')
        
        # Create xarray objects for speed and direction
        speed_da = xra.DataArray(
            speed,
            dims=u_wind.dims,
            coords=u_wind.coords,
            attrs={'units': 'knots', 'long_name': 'Wind speed'},
            name=conventions.WIND_SPEED
        )
        
        
        direction_da = xra.DataArray(
            direction,
            dims=u_wind.dims,
            coords=u_wind.coords,
            attrs={'units': 'radians', 'long_name': 'Wind direction'},
            name=conventions.WIND_DIRECTION
        )
        
        # Apply compression using existing functions
        print("Applying Beaufort scale compression...")
        speed_compressed = to_beaufort(speed_da)
        
        print("Applying integer direction compression...")
        direction_compressed = to_integer_direction(direction_da)
        
        # Create compressed data arrays with preserved chunking
        ds_compressed[conventions.WIND_SPEED] = xra.DataArray(
            speed_compressed[1],  # data
            dims=speed_compressed[0],  # dims
            coords={coord: ds_compressed.coords[coord] for coord in speed_compressed[0] if coord in ds_compressed.coords},
            attrs=speed_compressed[2],  # attrs
            name=conventions.WIND_SPEED
        )
        # Preserve chunking from original u_wind
        ds_compressed[conventions.WIND_SPEED] = ds_compressed[conventions.WIND_SPEED].chunk(u_wind.chunks)
        
        ds_compressed[conventions.WIND_DIRECTION] = xra.DataArray(
            direction_compressed[1],  # data
            dims=direction_compressed[0],  # dims
            coords={coord: ds_compressed.coords[coord] for coord in direction_compressed[0] if coord in ds_compressed.coords},
            attrs=direction_compressed[2],  # attrs
            name=conventions.WIND_DIRECTION
        )
        # Preserve chunking from original u_wind
        ds_compressed[conventions.WIND_DIRECTION] = ds_compressed[conventions.WIND_DIRECTION].chunk(u_wind.chunks)
        
        # Remove original wind components
        ds_compressed = ds_compressed.drop_vars([u_name, v_name])
    
    else:
        print("Warning: No wind components found, skipping wind compression")
    
    # Final cleanup: remove any non-dimensional coordinates that might cause zarr issues
    coords_to_keep = []
    for coord_name in ds_compressed.coords:
        if coord_name in ds_compressed.dims:
            coords_to_keep.append(coord_name)
        else:
            print(f"Dropping non-dimensional coordinate: {coord_name}")
    
    # Keep only coordinates that are also dimensions
    ds_compressed = ds_compressed.reset_coords(drop=True)
    
    print(f"Final coordinates: {list(ds_compressed.coords.keys())}")
    print(f"Final dimensions: {dict(ds_compressed.dims)}")
    print(f"Using Beaufort scale and integer direction compression")
    
    return ds_compressed


def process_and_store_wind_data(source_name=DEFAULT_SOURCE, output_path=None, variables=None):
    """
    Main function to fetch wind data, convert to speed/direction, compress, and store.
    
    Parameters:
    -----------
    source_name : str
        Name of the zarr source to use
    output_path : str
        Path where to store the processed zarr archive
    variables : list
        List of variables to fetch
    
    Returns:
    --------
    str
        Path to the stored zarr archive
    """
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
        output_path = f"./data/latest_forecast.zarr"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    intermediate_path = output_path.replace(".zarr", "_intermediate.zarr")
    print("Step 1: Fetching and converting wind data...")

    if os.path.isdir(intermediate_path):
        print(f"Intermediate path {intermediate_path} already exists. Skipping step 1.")
        wind_ds = xra.open_zarr(intermediate_path)
    else:
        wind_ds = fetch_wind_data(source_name, variables)

        with ProgressBar():
            # Create encoding that explicitly disables all compression and filters
            encoding = {}
            for var in wind_ds.data_vars:
                encoding[var] = {
                    'compressor': None,
                    'filters': None
                }
            # Also encode coordinates to disable compression
            for coord in wind_ds.coords:
                encoding[coord] = {
                    'compressor': None,
                    'filters': None
                }
            wind_ds.to_zarr(intermediate_path, mode='w', encoding=encoding)

    print("Step 2: Compressing wind data...")
    ds_compressed = compress_wind_data(wind_ds)

    print(f"Step 3: Storing as zarr {output_path} (completely uncompressed)")
    with ProgressBar():
        # Create encoding that explicitly disables all compression and filters
        encoding = {}
        for var in ds_compressed.data_vars:
            encoding[var] = {
                'compressor': None,
                'filters': None
            }
        # Also encode coordinates to disable compression
        for coord in ds_compressed.coords:
            encoding[coord] = {
                'compressor': None,
                'filters': None
            }
        ds_compressed.to_zarr(output_path, mode='w', encoding=encoding)

    print(f"Successfully processed and stored wind data at: {output_path}")
    
    # Print some statistics using standard coordinate names
    print("\nDataset Statistics:")
    if 'ensemble_member' in ds_compressed.dims:
        print(f"  Ensemble members: {ds_compressed.dims['ensemble_member']}")
    if 'lead_time' in ds_compressed.dims:
        print(f"  Lead times: {ds_compressed.dims['lead_time']}")
    print(f"  Latitude range: {ds_compressed[conventions.LATITUDE].min().values:.2f} to {ds_compressed[conventions.LATITUDE].max().values:.2f}")
    print(f"  Longitude range: {ds_compressed[conventions.LONGITUDE].min().values:.2f} to {ds_compressed[conventions.LONGITUDE].max().values:.2f}")
        
    return output_path


def main():
    """
    Main function to run the zarr-based wind data processing.
    """
    print("Starting zarr-based wind data processing...")
    
    # Process command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process wind data from zarr sources')
    parser.add_argument('--source', default=DEFAULT_SOURCE, 
                       choices=list(ZARR_SOURCES.keys()),
                       help='Zarr source to use')
    parser.add_argument('--output', default=None,
                       help='Output path for the zarr archive')    
    args = parser.parse_args()
    
    # Process and store the wind data using the specified zarr version
    zarr_path = process_and_store_wind_data(
            source_name=args.source,
            output_path=args.output,
        )
    
    print(f"\nProcessing complete! Zarr archive stored at: {zarr_path}")
    
    # Optionally, verify the stored data
    print("\nVerifying stored data...")
    ds_verify = xra.open_zarr(zarr_path)
    print(f"Stored dataset shape: {ds_verify.dims}")
    print(f"Stored variables: {list(ds_verify.data_vars.keys())}")


if __name__ == "__main__":
    main()
