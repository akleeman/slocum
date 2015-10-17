==================================
slocum
==================================
-------------------
*Weather forecasts for the bandwidth impaired*
-------------------


What is slocum?
~~~~~~~~~~~~~~~

slocum is a service that provides weather forecasts using a unique, weather specific, compression algorithm.
It includes an e-mail based forecast service similar to `saildocs <http://www.saildocs.com/>`_
and `zygrib <http://www.zygrib.org/>`_ with a few notable differences:

* **Forecast files are 10% smaller.** slocum doesn't waste a byte.  Each forecast is compressed using a weather specific compression algorithm that results in forecast files that are a fraction of the size of GRIB files.
* **Forecasts include probability.** Weather forecasts aren't perfect.  You know that.  We know that.  By providing access to ensemble forecasts, slocum helps you understand how much you can trust a given forecast.

Contents:

.. toctree::
   :maxdepth: 2

	intro
