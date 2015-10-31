Spot Forecasts
==============

Gridded forecasts hold forecasts for an entire region.  These are essentially
equivalent to the grib forecast requests from other providers, but may actually
consist of an entire `ensemble <ensembles.html>`_ of forecasts instead of a single forecast.

Here for example is a gridded ensemble forecast for the area around San Francisco.

.. plot::
    :height: 150
    :width: 600
    :align: center

    import xray
    import matplotlib.pyplot as plt
    from slocum import visualize
    fcsts = xray.open_dataset('./sf_example_forecast.nc')
    one_loc = fcsts.isel(latitude=5, longitude=5)
    plt.figure(figsize=(12, 7))
    visualize.plot_forecast(one_loc)
    plt.show()

