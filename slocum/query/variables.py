import numpy as np

from slocum.compression import schemes


class Latitude(schemes.SmallCoordinate):

    def __init__(self):
        super(Latitude, self).__init__(variable_name='latitude',
                                       units='degrees north',
                                       least_significant_digit=3)

    def normalize(self, ds):
        ds = ds.copy(deep=True)
        # make sure latitudes are in degrees and are on the correct scale
        assert 'degrees' in ds[self.variable_name].attrs['units']
        assert np.min(np.asarray(ds[self.variable_name].values)) >= -90
        assert np.max(np.asarray(ds[self.variable_name].values)) <= 90
        return ds


class Longitude(schemes.SmallCoordinate):

    def __init__(self):
        super(Longitude, self).__init__(variable_name='longitude',
                                       units='degrees east',
                                       least_significant_digit=3)

    def normalize(self, ds):
        ds = ds.copy(deep=True)
        # make sure longitudes are in degrees and are on the correct scale
        assert 'degrees' in ds[self.variable_name].attrs['units']
        wrap = lambda x: np.mod(x + 180., 360) - 180.
        ds[self.variable_name] = wrap(ds[self.variable_name].values)
        return ds

time = schemes.TimeCoordinate('time')

realization = schemes.TrivialCoordinate('realization')

temperature = schemes.SmallVariable('air_temperature', units='c',
                                   least_significant_digit = 0)

sea_surface_temperature = schemes.SmallVariable('sea_surface_temperature',
                                    units='c',
                                    least_significant_digit = 0)

# this pressure scale was derived by taking all the MSL pressures
# from a forecast run and computing the quantiles (then rounding to
# more friendly numbers).  We might find that we can add precision
# by focusing the bins around pressures expected in sailing waters
# rather than globally ... but for now this should work.
pressure = schemes.TinyVariable('air_pressure_at_sea_level',
                        units='Pa',
                        bins=np.array([97500., 99000., 99750,
                                       100500., 100700., 100850.,
                                       101000., 101150., 101350.,
                                       101600., 101900., 102150.,
                                       102500., 103100., 104000.]))

# Wind uses the beaufort scale for compression
wind_bins = np.array([0., 1., 3., 6., 10., 16., 21., 27.,
                       33., 40., 47., 55., 63., 75.]) / 1.94384449
wind_bin_names = ['F-{0:<12}'.format(i) for i in range(wind_bins.size - 1)]
wind = schemes.VelocityVariable(u_name='x_wind',
                                v_name='y_wind',
                                variable_name='wind',
                                speed_bins=wind_bins,
                                speed_bin_names=wind_bin_names)

# Current uses a vaguely logarithmic scale in knots.
current_bins = np.array([0., 0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
                          0.7, 1., 1.5, 2., 3., 4., 5., 7.]) / 1.94384449
current = schemes.MaskedVelocity(u_name='sea_water_x_velocity',
                                 v_name='sea_water_y_velocity',
                                 variable_name="sea_water",
                                 speed_bins=current_bins,
                                 direction_orientation='to')

wave_direction = schemes.TinyDirection('sea_surface_wave_to_direction')
wave_height = schemes.TinyVariable('sea_surface_wave_significant_height',
                                   units='m',
                                   bins=np.array([0., 0.2, 0.5, 0.75, 1., 1.5, 2.,
                                                  3., 4., 5., 6., 7., 8., 10., 15.]))
wave = schemes.CombinedVariable([wave_height, wave_direction])