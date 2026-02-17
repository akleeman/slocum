# API Service

FastAPI backend for ECMWF ensemble map products and click-detail queries.

## Run locally

```bash
pixi run api-dev
```

## Test

```bash
pixi run api-test
```

## Endpoints

- `GET /meta/cycles`
- `GET /tiles/temperature_2m_member/{cycle}/{lead_hour}/{z}/{x}/{y}`
- `GET /point/wind?cycle=&lead=&lat=&lon=`
- `GET /timeseries/wind?cycle=&lat=&lon=`
- `GET /timeseries/wind/ensembles?cycle=&lat=&lon=`
