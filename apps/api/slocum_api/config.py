from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    dataset_bucket: str = "dynamical-ecmwf-ifs-ens"
    dataset_prefix: str = "ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/"
    dataset_region: str = "us-west-2"
    dataset_branch: str = "main"
    max_cycle_count: int = 14
    tile_target_px: int = 64
    request_timeout_seconds: float = 20.0
    transient_read_retries: int = 2
    transient_backoff_seconds: float = 0.5


def get_settings() -> Settings:
    return Settings(
        dataset_bucket=os.getenv("SLOCUM_DATASET_BUCKET", Settings.dataset_bucket),
        dataset_prefix=os.getenv("SLOCUM_DATASET_PREFIX", Settings.dataset_prefix),
        dataset_region=os.getenv("SLOCUM_DATASET_REGION", Settings.dataset_region),
        dataset_branch=os.getenv("SLOCUM_DATASET_BRANCH", Settings.dataset_branch),
        max_cycle_count=int(os.getenv("SLOCUM_MAX_CYCLE_COUNT", str(Settings.max_cycle_count))),
        tile_target_px=int(os.getenv("SLOCUM_TILE_TARGET_PX", str(Settings.tile_target_px))),
        request_timeout_seconds=float(
            os.getenv("SLOCUM_REQUEST_TIMEOUT_SECONDS", str(Settings.request_timeout_seconds))
        ),
        transient_read_retries=int(
            os.getenv("SLOCUM_TRANSIENT_READ_RETRIES", str(Settings.transient_read_retries))
        ),
        transient_backoff_seconds=float(
            os.getenv("SLOCUM_TRANSIENT_BACKOFF_SECONDS", str(Settings.transient_backoff_seconds))
        ),
    )
