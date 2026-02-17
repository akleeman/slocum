.. image:: https://travis-ci.org/akleeman/slocum.svg?branch=master
    :target: https://travis-ci.org/akleeman/slocum

Slocum -- A tool for ocean passage planning
===========================================

**slocum** is an open source project which provides weather forecasts
to the bandwidth impaired (primarily sailors).  It includes a weather
specific compression algorithm which results in forecasts attachments
that are the fraction of the size of `saildoc <http://www.saildocs.com/>`_
forecasts.

Beta Web Version: http://www.ensembleweather.com/

Why the name **slocum**?
------------------------

Joshua Slocum (February 20, 1844 -on or shortly after November 14, 1909)
was a Canadian-American seaman and adventurer, a noted writer, and the first
man to sail single-handedly around the world. In 1900 he told the story of
this in Sailing Alone Around the World. He disappeared in November 1909
while aboard his boat, the Spray. (wikipedia)

In "Sailing Alone Around the World" he talks about his tin clock:

  "My tin clock and only timepiece had by this time lost its minute-hand, 
   but after I boiled her she told the hours, and that was near enough on
   a long stretch." 

If Slocum was able to successfully circumnavigate via celestial navigation
using a clock with no minute hand, then modern day sailors should be able
to get by using weather forecasts without floating point precision.


Modern Web Stack MVP (Track A)
------------------------------

The repository now includes a new weather web workflow:

- **Backend:** Python/FastAPI in ``apps/api`` using ``xarray`` and Icechunk/Zarr.
- **Frontend:** TypeScript/React in ``apps/web`` using MapLibre + deck.gl.

Local prerequisites
~~~~~~~~~~~~~~~~~~~

- Pixi (for Python environment management)
- Node.js 20+

Install and run locally
~~~~~~~~~~~~~~~~~~~~~~~

::

  pixi install
  pixi run api-runtime-deps
  pixi run web-install
  pixi run dev-local

This starts:

- API at ``http://127.0.0.1:8000``
- Web app at ``http://127.0.0.1:5173``

Test commands
~~~~~~~~~~~~~

::

  pixi run api-test
  pixi run web-test
  pixi run web-e2e

Current API endpoints (MVP + click details)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~- ``GET /meta/cycles``
- ``GET /tiles/temperature_2m_member/{cycle}/{lead_hour}/{z}/{x}/{y}``
- ``GET /point/wind?cycle=&lead=&lat=&lon=``
- ``GET /timeseries/wind?cycle=&lat=&lon=``
- ``GET /timeseries/wind/ensembles?cycle=&lat=&lon=``

Production deployment baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Serve ``apps/web`` as static assets from CDN/object storage.
- Run ``apps/api`` as a containerized FastAPI service.
- Keep dataset access read-only against Icechunk/S3 source.
- Tile endpoints emit cache headers for edge caching.