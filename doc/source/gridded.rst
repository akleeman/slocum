Gridded Forecasts
==============

Gridded forecasts hold forecasts for an entire region.  These are essentially
equivalent to the grib forecast requests from other providers, but may actually
consist of an entire :doc:`ensemble <ensembles>` of forecasts instead of a single forecast.

For example, here is a gridded ensemble forecast for the area around San Francisco.  Each
of the circles in the plot below show the outcome from 21 different forecasts for that
point and time.  This lets you get an understanding of not only what the most likely
forecast is, but how much you trust the forecast.  See the section on :ref:`visualizing ensemble forecasts <visualize-ensemble>`
for more details.

.. plot::
    :align: center

    import xarray as xra
    import matplotlib.pyplot as plt
    from slocum import visualize
    fcsts = xray.open_dataset('./sf_example_forecast.nc')
    fig, axis = plt.subplots(1, 1, figsize=(8, 8))

    fcst = fcsts.isel(time=[1])
    visualize.plot_gridded_forecast(fcst, ax=axis)
    axis.set_title("6 hour forecast")

    plt.show()

