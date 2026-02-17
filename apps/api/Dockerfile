FROM mambaorg/micromamba:1.5.10

WORKDIR /opt/app
COPY pixi.toml pixi.lock setup.py README.rst /opt/app/
COPY slocum /opt/app/slocum
COPY apps/api /opt/app/apps/api

RUN micromamba install -y -n base -c conda-forge \
    python=3.11 fastapi uvicorn xarray numpy scipy pandas pyproj cftime zarr requests arraylake && \
    pip install -e . && \
    micromamba clean --all --yes

EXPOSE 8000
CMD ["uvicorn", "slocum_api.main:app", "--app-dir", "apps/api", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
