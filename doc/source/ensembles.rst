.. _ensembles:

Ensemble Forecasts
===========================

What are ensemble forecasts?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Ensemble forecasting <https://en.wikipedia.org/wiki/Ensemble_forecasting>`_ is commonly used to help
understand how much you can trust a given weather forecast.  The concept is relatively simple, a numerical weather
model (such as GFS) is run multiple times using slightly different inputs.  The result is not just one
forecast, but an entire ensemble of forecasts.  You can then look at how much the forecasts agree to
get a sense of how much you can trust them.

Take this forecast for a location off the coast of San Francisco as an example.  The plot shows
21 forecasts overlayed on top of each other,

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

Each grid box is colored to show the number of forecasts that
fell in the indicated range.  The darker the color the larger the number of the forecasts.
For example, on October 19th at 06:00 (the very first forecast period) all the forecasts fall
in the 16-21 knot range (resulting in a solid dark blue color) and suggesting that you can
trust the forecast.  Later, at 18:00 that day, the forecasts disagree showing
wind speeds in the 3-10 knot range (though possibly as low as 1 knot and as high as 16).
Similarly the forecast further out, for October 24th at 00:00, shows strong forecast disagreement
with wind speeds anywhere from 3 to 33 knots.

A few caveats. Even ensemble forecasts aren't perfect. Each of the forecast members comes from
an numerical model and numerical models are never perfect. To truly understand the uncertainty
of a particular forecast you'd need to not only know the internal uncertainty (this is what
ensemble forecasts provide) but also the model uncertainty -- i.e, what is it that the model
is not capable of producing.

Visualizing Ensemble Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One challenge with ensemble forecasts is that it's difficult to show multiple forecasts at the
same time.  Doing so with a standard wind barb plot would get confusing.  Instead ``slocum``
uses a combination of probabilistic line plots (like the one above) and what we call
a "wind circle" which represents an ensemble of wind speeds and directions at a single location and time.
Below are a few examples of wind circles:

.. plot::
    :height: 100

    import numpy as np
    import matplotlib.pyplot as plt

    from slocum.visualize import velocity
    from slocum.query.variables import wind

    n = 21
    np.random.seed(1982)
    north = np.zeros(n)
    north_and_west = -np.pi / 2 * (np.arange(n) > n - 2)
    all_over = np.random.randint(16, size=21) * np.pi / 8

    fig, axes = plt.subplots(1, 4, figsize=(10, 2))

    for ax, dir in zip(axes.reshape(-1),
                       [north, north_and_west, all_over]):
        ax.set_axis_off()
        velocity.DirectionCircle(0.5, 0.5,
                                 np.ones(n), dir, 0.5,
                                 norm=plt.Normalize(-1, 0),
                                 ax=ax)
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlim([-0.1, 1.1])

    norm = plt.cm.colors.BoundaryNorm(wind.speed_bins, wind.speed_bins.size)
    speeds = np.linspace(6., 15, n)
    dirs = np.linspace(0., np.pi/2, n)
    axes[-1].set_axis_off()
    velocity.DirectionCircle(0.5, 0.5,
                             speeds, dirs, 0.5,
                             cmap=velocity.velocity_cmap,
                             norm=norm,
                             ax=axes[-1])
    axes[-1].set_ylim([-0.1, 1.1])
    axes[-1].set_xlim([-0.1, 1.1])
    plt.show()

From left to right we see several different situations.  The farthest left
shows the case where all the wind forecasts are from the north.  The next shows wind mostly from the
north with a small chance of westerly winds.  Next we see a forecast where the wind direction is
completely random which suggests that the forecast should not be trusted. Finally we have a color
coded wind circle used when showing gridded forecasts.  In this case the forecast shows that the
wind speed may vary from Force 4 (green) to Force 6 (orange) and that the stronger wind speeds
are more likely to come from the East.

Here are two examples of how wind circles are used to show ensemble forecasts for northern
California.  The first forecast is for six hours into the forecast period, the second is
for 120 hours into the forecast period.

.. plot::
    :align: center

    import xray
    import matplotlib.pyplot as plt
    from slocum import visualize
    fcsts = xray.open_dataset('./sf_example_forecast.nc')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    first = fcsts.isel(time=[1])
    visualize.plot_gridded_forecast(first, ax=axes[0])
    axes[0].set_title("6 hour forecast")

    ind = 20
    later = fcsts.isel(time=[20])
    visualize.plot_gridded_forecast(later, ax=axes[1])
    tdiff = (fcsts['time'].values[ind] - fcsts['time'].values[0]).astype('timedelta64[h]')
    axes[1].set_title("%d hour forecast" % tdiff.astype('int'))
    plt.show()

The six hour forecast is reasonably easy to interpret, winds will generally be coming from the north/west
with speeds around 20 knots.  It gets more difficult to interpret the 120 hour forecast, but that's due
mostly to the fact that the forecasts do not agree as much.
