Query Syntax
============

There are two different types of queries you can make using slocum, :doc:`gridded <gridded>` and
:doc:`spot <spot>`.  Gridded forecasts contain a forecast for an entire region while spot forecasts
are for a single location only.  The query syntax for the two are slightly
different and follow the saildocs syntax.

Query Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

Slocum uses a query syntax that is similar to `saildocs <http://www.saildocs.com/gribinfo>`_ or
`zygrib <http://www.zygrib.org/?page=gribauto>`_. If you're familiar with these, using slocum will be easy.

For a full description of the possible queries see below, but here are a few examples:

``send GEFS:35N,45N,120W,130W|native|0,6..144|WIND``  will send an :doc:`ensemble <ensembles>` of 21 :doc:`gridded forecasts <gridded>`
from the GEFS (Global Ensemble Forecast System) model which covers a ten by ten degree area around San Francisco.
The forecast will use the native (1 degree) resolution and contain a wind forecast every six hours for the next six days.

``send GFS:35N,45N,120W,130W|2,2|0,3,6,12,24|WIND``  will send a single :doc:`gridded forecast <gridded>`
from the GFS model which covers a ten by ten degree area around San Francisco.  The forecast will be 2 degrees
resolution and contain a wind forecast for the next 3, 6, 12 and 24 hours.

``send spot:32N,-117E|8,6|wind`` will send a `spot forecast <spot.html>`_ just off the coast of San Diego that
has a wind forecast for every six hours for the next eight days.

Simply include one (or all) of those queries in an email body and send it to query@ensembleweather.com like this:

::

    to: query@ensembleweather.com
    subject: send me some forecasts please
    body:
    
    send GEFS:35N,45N,120W,130W|native|0,6..144|WIND
    send GFS:35N,45N,120W,130W|1,1|0,3,6,12,24|WIND
    send spot:32N,-117E|8,6|wind


Gridded Forecasts
~~~~~~~~~~~~~~~~~~

A gridded query requires specifying the forecast model, region, resolution, forecast times and variables
which can be done as follows:

.. include:: gridded_usage.rst

Spot Forecasts
~~~~~~~~~~~~~~~~

Spot forecasts follow a slightly different format, namely:

.. include:: spot_usage.rst
