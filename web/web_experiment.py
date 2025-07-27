"""
Web experiment for processing wind data from zarr sources.

This module uses the slocum conventions module to ensure consistent
variable naming and coordinate handling across different data sources.
"""

import os
import shutil
import warnings
import zarr

import numpy as np
import xarray as xra
import multiprocessing as mp

from itertools import chain
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


def find_latest_zarr_source(source_name=DEFAULT_SOURCE):
    """
    Find the most recent zarr dataset from the specified source.
    For now, we'll use a known working zarr endpoint.
    """
    source_url = ZARR_SOURCES.get(source_name)
    if not source_url:
        raise ValueError(f"Unknown source: {source_name}")
    
    return source_url


def fetch_and_convert_wind_data(source_name=DEFAULT_SOURCE, variables=None):
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
    
    # Open the zarr dataset with chunking to keep it lazy
    ds = xra.open_zarr(zarr_url, decode_timedelta=True, chunks='auto')

    # get latest
    ds = ds.isel(init_time=-1)
    
    # Select only the variables we need
    available_vars = [var for var in variables if var in ds]
    if not available_vars:
        print(f"Available variables: {list(ds.variables.keys())}")
        print(f"Requested variables {variables} not found.")
        
        # Try to find wind components using conventions
        if conventions.has_wind_components(ds):
            u_name, v_name = conventions.get_wind_component_names(ds)
            available_vars = [u_name, v_name]
            print(f"Found wind components using conventions: {available_vars}")
        else:
            raise ValueError("No wind components found in dataset")
    
    print(f"Available variables: {list(ds.variables.keys())}")
    print(f"Fetching variables: {available_vars}")
    
    # Select the variables
    ds_subset = ds[available_vars]
    
    # Rename variables to standard slocum names
    ds_subset = conventions.rename_variables_to_standard(ds_subset)
    
    print(f"Dataset shape: {ds_subset.dims}")
    print(f"Dataset variables: {list(ds_subset.data_vars.keys())}")
    
    # Convert to wind speed and direction
    u_wind_name, v_wind_name = conventions.get_wind_component_names(ds_subset)
    speed, direction = vector_to_radial(ds_subset[u_wind_name], ds_subset[v_wind_name], orientation='from')
    
    # Convert speed to knots (lazy version)
    speed.attrs['units'] = 'm s**-1'
    
    # Manual unit conversion to avoid forcing computation
    # 1 m/s = 1.9438444924574 knots
    speed_knots = speed * 1.9438444924574
    speed_knots.attrs = speed.attrs.copy()
    speed_knots.attrs['units'] = 'knots'
    
    # Create new dataset with speed and direction
    ds_wind = xra.Dataset({
        conventions.WIND_SPEED: speed_knots,
        conventions.WIND_DIRECTION: direction
    })

    print("Dataset created - operations remain lazy until .compute() or zarr write")
    
    # Copy only coordinates that are also dimensions (avoid metadata coords)
    dimensional_coords = {}
    for coord_name in ds_subset.coords:
        if coord_name in ds_subset.dims:
            dimensional_coords[coord_name] = ds_subset[coord_name]
    
    # Add dimensional coordinates to the dataset
    for coord_name, coord_data in dimensional_coords.items():
        ds_wind[coord_name] = coord_data
    
    print(f"Kept dimensional coordinates: {list(ds_wind.coords.keys())}")
    print(f"Dropped {len(ds_subset.coords) - len(dimensional_coords)} non-dimensional coordinates")
        
    return ds_wind


def to_beaufort(knots):
    """Convert wind speeds to Beaufort scale (lazy version)."""
    import dask.array as da
    
    # Work with dask arrays to maintain laziness
    if hasattr(knots, 'data') and hasattr(knots.data, 'map_blocks'):
        # This is a dask-backed xarray
        def _digitize_block(block):
            return np.digitize(block, wind_bins).astype('u8')
        
        beaufort_data = knots.data.map_blocks(
            _digitize_block, 
            dtype='u8', 
            drop_axis=None
        )
    else:
        # Fallback to immediate computation if not dask-backed
        beaufort_data = np.digitize(knots.values, wind_bins).astype('u8')
    
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
            return ints.astype('u8')
        
        direction_data = direction.data.map_blocks(
            _digitize_direction_block,
            dtype='u8',
            drop_axis=None
        )
    else:
        # Fallback to immediate computation if not dask-backed
        ints = np.digitize(direction.values, dir_bins) % len(dir_bins)
        direction_data = ints.astype('u8')
    
    attrs = direction.attrs.copy()
    return (direction.dims, direction_data, attrs)


def compress_wind_data(ds):
    """
    Apply compression to wind data (Beaufort scale for speed, integer bins for direction).
    """
    print("Before compression:")
    print(f"  Coordinates: {list(ds.coords.keys())}")
    print(f"  Data variables: {list(ds.data_vars.keys())}")
    print(f"  Dimensions: {dict(ds.dims)}")
    
    ds_compressed = ds.copy()
    
    # Reduce latitude domain to exclude poles using standard coordinate name
    ds_compressed = ds_compressed.sel(**{conventions.LATITUDE: slice(80, -80)})
    
    # Apply Beaufort scale compression to wind speeds
    if conventions.WIND_SPEED in ds_compressed:
        dims, data, attrs = to_beaufort(ds_compressed[conventions.WIND_SPEED])
        ds_compressed[conventions.WIND_SPEED] = xra.DataArray(
            data,  dims=dims, attrs=attrs, name=conventions.WIND_SPEED
        )
    
    # Apply integer direction compression
    if conventions.WIND_DIRECTION in ds_compressed:
        dims, data, attrs = to_integer_direction(ds_compressed[conventions.WIND_DIRECTION])
        ds_compressed[conventions.WIND_DIRECTION] = xra.DataArray(
            data, dims=dims, attrs=attrs, name=conventions.WIND_DIRECTION
        )
    
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
    
    return ds_compressed


def store_as_zarr_v3(ds, output_path):
    """
    Store the dataset as a zarr v3 archive using sharding for optimal performance.
    Optimize for lat/lon access across lead times while keeping all ensemble members.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset to store
    output_path : str
        Path where to store the zarr archive
    """
    # Remove existing zarr store if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    print(f"Storing dataset as zarr v3 with sharding at: {output_path}")
    
    # Define sharding strategy optimized for lat/lon access across lead times
    # Shards are larger containers, chunks are smaller units within shards
    encoding = {}
    for var in ds.data_vars:
        var_dims = ds[var].dims
        shards = []
        chunks = []
        
        for dim in var_dims:
            if dim == 'ensemble_member':
                shards.append(ds.sizes[dim])  # Keep all ensemble members in one shard
                chunks.append(1)  # Small chunks for individual ensemble access
            elif dim == 'lead_time':
                shards.append(min(ds.sizes[dim], 50))  # Moderate shard size for time
                chunks.append(5)  # Small time chunks for efficient time series access
            elif dim == conventions.LATITUDE:
                shards.append(200)  # Larger spatial shards
                chunks.append(50)   # Smaller spatial chunks for random access
            elif dim == conventions.LONGITUDE:
                shards.append(200)  # Larger spatial shards  
                chunks.append(50)   # Smaller spatial chunks for random access
            else:
                shards.append(1)
                chunks.append(1)
        
        encoding[var] = {
            'shards': tuple(shards),
            'chunks': tuple(chunks),
            'dtype': 'uint8'
        }
    
    print(f"Using zarr v3 sharding configuration:")
    for var, enc in encoding.items():
        if 'shards' in enc:
            print(f"  {var}: shards={enc['shards']}, chunks={enc['chunks']}")
    
    with ProgressBar():
        # Store with zarr format v3 and sharding - let zarr handle the rechunking during write
        # This avoids creating a huge dask task graph in memory
        print("Writing to zarr - zarr will handle rechunking during write to avoid memory issues")
        ds.to_zarr(
            output_path, 
            mode='w',
            zarr_format=3,
            encoding=encoding,
            safe_chunks=False  # Allow zarr to rechunk during write
        )
    
    print(f"Successfully stored zarr v3 archive at: {output_path}")
    return output_path


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
        output_path = f"./data/processed_wind_{timestamp}.zarr"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Step 1: Fetching and converting wind data...")
    ds_wind = fetch_and_convert_wind_data(source_name, variables)

    print("Step 2: Compressing wind data...")
    ds_compressed = compress_wind_data(ds_wind)
    
    print(f"Compressed dataset shape: {ds_compressed.dims}")
    
    with ProgressBar():
        print("Step 3: Storing as zarr archive...")
        zarr_path = store_as_zarr_v3(ds_compressed, output_path)
    
    print(f"Successfully processed and stored wind data at: {zarr_path}")
    
    # Print some statistics using standard coordinate names
    print("\nDataset Statistics:")
    if 'ensemble_member' in ds_compressed.dims:
        print(f"  Ensemble members: {ds_compressed.dims['ensemble_member']}")
    if 'lead_time' in ds_compressed.dims:
        print(f"  Lead times: {ds_compressed.dims['lead_time']}")
    print(f"  Latitude range: {ds_compressed[conventions.LATITUDE].min().values:.2f} to {ds_compressed[conventions.LATITUDE].max().values:.2f}")
    print(f"  Longitude range: {ds_compressed[conventions.LONGITUDE].min().values:.2f} to {ds_compressed[conventions.LONGITUDE].max().values:.2f}")
    
    if conventions.WIND_SPEED in ds_compressed:
        print(f"  Wind speed range: {ds_compressed[conventions.WIND_SPEED].min().values} to {ds_compressed[conventions.WIND_SPEED].max().values} (Beaufort scale)")
    
    return zarr_path


def process_and_store_wind_data_v3(source_name=DEFAULT_SOURCE, output_path=None, variables=None):
    """
    Process wind data using zarr v3 with sharding - keeps all ensemble members and lead times.
    
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
        output_path = f"./data/processed_wind_v3_{timestamp}.zarr"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Step 1: Fetching and converting wind data...")
    ds_wind = fetch_and_convert_wind_data(source_name, variables)
    
    print("Step 2: Compressing wind data...")
    ds_compressed = compress_wind_data(ds_wind)
    
    print(f"Compressed dataset shape: {ds_compressed.dims}")
    
    with ProgressBar():
        print("Step 3: Storing as zarr v3 archive with sharding...")
        zarr_path = store_as_zarr_v3(ds_compressed, output_path)
    
    print(f"Successfully processed and stored wind data at: {zarr_path}")
    
    # Print some statistics using standard coordinate names
    print("\nDataset Statistics:")
    if 'ensemble_member' in ds_compressed.dims:
        print(f"  Ensemble members: {ds_compressed.dims['ensemble_member']}")
    if 'lead_time' in ds_compressed.dims:
        print(f"  Lead times: {ds_compressed.dims['lead_time']}")
    print(f"  Latitude range: {ds_compressed[conventions.LATITUDE].min().values:.2f} to {ds_compressed[conventions.LATITUDE].max().values:.2f}")
    print(f"  Longitude range: {ds_compressed[conventions.LONGITUDE].min().values:.2f} to {ds_compressed[conventions.LONGITUDE].max().values:.2f}")
    
    if conventions.WIND_SPEED in ds_compressed:
        print(f"  Wind speed range: {ds_compressed[conventions.WIND_SPEED].min().values} to {ds_compressed[conventions.WIND_SPEED].max().values} (Beaufort scale)")
    
    return zarr_path


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
    parser.add_argument('--variables', nargs='+', default=None,
                       help='Variables to fetch from the source')
    parser.add_argument('--zarr-version', choices=['2', '3'], default='2',
                       help='Zarr format version to use (default: 2)')
    
    args = parser.parse_args()
    
    # Process and store the wind data using the specified zarr version
    if args.zarr_version == '3':
        zarr_path = process_and_store_wind_data_v3(
            source_name=args.source,
            output_path=args.output,
            variables=args.variables
        )
    else:
        zarr_path = process_and_store_wind_data(
            source_name=args.source,
            output_path=args.output,
            variables=args.variables
        )
    
    print(f"\nProcessing complete! Zarr archive stored at: {zarr_path}")
    
    # Optionally, verify the stored data
    print("\nVerifying stored data...")
    ds_verify = xra.open_zarr(zarr_path)
    print(f"Stored dataset shape: {ds_verify.dims}")
    print(f"Stored variables: {list(ds_verify.data_vars.keys())}")


if __name__ == "__main__":
    main()
