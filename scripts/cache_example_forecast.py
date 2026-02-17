from __future__ import annotations

import argparse
from pathlib import Path

import icechunk
import xarray as xr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a small local Zarr subset for fast visual/test iteration.")
    parser.add_argument("--bucket", default="dynamical-ecmwf-ifs-ens")
    parser.add_argument(
        "--prefix",
        default="ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/",
    )
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--output", default=str(Path.home() / "data" / "example_forecast.zarr"))
    parser.add_argument("--lead-count", type=int, default=8, help="Number of lead times from the start to store")
    parser.add_argument("--member-index", type=int, default=0, help="Ensemble member index to keep")
    parser.add_argument("--west", type=float, default=-150.0)
    parser.add_argument("--east", type=float, default=-110.0)
    parser.add_argument("--south", type=float, default=20.0)
    parser.add_argument("--north", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    storage = icechunk.s3_storage(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(args.branch)
    ds = xr.open_zarr(session.store, chunks="auto")

    latest_cycle = ds["init_time"].values[-1]
    latest = ds.sel(init_time=latest_cycle).isel(
        lead_time=slice(0, args.lead_count),
        ensemble_member=args.member_index,
    )

    lats = latest["latitude"]
    lat_slice = (
        slice(args.north, args.south) if bool((lats[0] > lats[-1]).item()) else slice(args.south, args.north)
    )
    latest = latest.sel(latitude=lat_slice, longitude=slice(args.west, args.east))

    # Keep only fields needed for current map + click details workflows.
    keep_vars = ["temperature_2m", "wind_u_10m", "wind_v_10m"]
    latest = latest[keep_vars]

    if output.exists():
        import shutil

        shutil.rmtree(output)
    latest.to_zarr(output, mode="w")
    print(f"Wrote cached subset to {output}")
    print(f"cycle={str(latest_cycle)} dims={dict(latest.sizes)} vars={list(latest.data_vars)}")


if __name__ == "__main__":
    main()
