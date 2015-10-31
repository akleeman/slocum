Query Syntax
============

There are two different types of queries you can make using slocum, gridded and
`spot <spot.html>`_.  Gridded forecasts contain a forecast for an entire region while spot forecasts
are for a single location only.  The query syntax for the two are slightly
different and follow the saildocs syntax.

Query Examples
~~~~~~~~~~~~~~~~~

``send GFS:35N,45N,120W,130W|1,1|0,6..144|WIND``  will send a single forecast from the GFS model which
covers a ten by ten degree area around San Francisco.  The forecast will be 1 degree resolution and contain
a wind forecast every six hours for the next six days.

``send spot:32N,-117E|8,6|wind`` will send a `spot forecast <spot.html>`_ just off the coast of San Diego that
has a wind forecast for every six hours for the next eight days.


Gridded Forecasts
~~~~~~~~~~~~~~~~~~

A gridded query requires specifying the forecast model, region, resolution, forecast times and variables
which can be done as follows:

.. include:: gridded_usage.rst

Spot Forecasts
~~~~~~~~~~~~~~~~

Spot forecasts follow a slightly different format, namely:

.. include:: spot_usage.rst
