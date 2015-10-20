import matplotlib.pyplot as plt

import slocum.query.utils as query_utils
from slocum.compression import schemes

import spot
import utils
import velocity
import interactive



def plot_spot(fcst, variable_name=None):
    """
    Takes a forecast that is assumed to hold a spot forecast and
    infers the type of variable then uses the corresponding plot utilities.
    """
    variable_name = variable_name or utils.infer_variable(fcst)
    variable = query_utils.get_variable(variable_name)
    if isinstance(variable, schemes.VelocityVariable):
        spot.spot_velocity(fcst, variable)
    else:
        raise NotImplementedError("Only support velocity variables at "
                                  "the moment.")


def plot_gridded_forecast(fcst, variable_name=None):
    """
    Creates an interactive plot of a gridded forecast.
    """
    variable_name = variable_name or utils.infer_variable(fcst)
    variable = query_utils.get_variable(variable_name)
    if isinstance(variable, schemes.VelocityVariable):
        velocity.VelocityField(fcst, variable)
    else:
        raise NotImplementedError("Only support velocity variables at "
                                  "the moment.")


def plot_interactive_forecast(fcst, variable_name=None):
    variable_name = variable_name or utils.infer_variable(fcst)
    variable = query_utils.get_variable(variable_name)
    if isinstance(variable, schemes.VelocityVariable):
        interactive.InteractiveVelocity(fcsts, velocity_variable)
    else:
        raise NotImplementedError("Only support velocity variables at "
                                  "the moment.")


def plot_forecast(fcst, variable_name=None):
    """
    Performs some inference on the forecast and decides which
    type of plot is most appropriate.
    """
    if utils.is_spot_forecast(fcst):
        plot_spot(fcst, variable_name)
        plt.show()
    elif fcst.dims['time'] == 1:
        plot_gridded_forecast(fcst, variable_name)
    else:
        plot_interactive_forecast(fcst, variable_name)
