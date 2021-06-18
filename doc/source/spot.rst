Spot Forecasts
==============

Spot Forecasts are forecasts for a single location.  ``slocum`` is capable of
sending the equivalent of 21 forecasts in a file the same size as a saildocs
SPOT forecast.  For a query like ``send spot:gefs:40N,125W|6,6|wind`` the
file will be about 1 kB and once plotted will look like this:

.. plot::
    :height: 150
    :width: 600
    :align: center

    import xarray as xra
    import matplotlib.pyplot as plt
    from slocum import visualize
    fcsts = xray.open_dataset('./sf_example_forecast.nc')
    one_loc = fcsts.isel(latitude=5, longitude=5)
    plt.figure(figsize=(12, 7))
    visualize.plot_forecast(one_loc)
    plt.show()

Don't want all 21 forecats?  You can get a single forecast using the same
query as you would for saildocs (``send spot:gfs:40N,125W|6,6|wind``).  In
this example the resulting file is about 500 bytes, or half the size of a saildocs SPOT
forecast.

.. plot::
    :height: 150
    :width: 600
    :align: center

    import xarray as xra
    import matplotlib.pyplot as plt
    from slocum import visualize
    fcsts = xray.open_dataset('./sf_example_forecast.nc')
    one_loc = fcsts.isel(latitude=5, longitude=5, realization=[0])
    plt.figure(figsize=(12, 7))
    visualize.plot_forecast(one_loc)
    plt.show()

Note however that the single forecast plot contains much less information and is only half
the size of the request with 21 forecasts.  Unless you are extremely bandwidth constrained
you'll probably want to use the full spot forecasts.
