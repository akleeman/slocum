import sys
import os.path
import zlib
import base64
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sl.lib import tinylib
import xray


def display_espot(fcsts, f_var, plot_cols):
    """
    Plots the spread of values for forecast variable f_var in an ensemble
    forecast along the forecast times contained in fcsts, a list of
    dictionaries with members of the ensemble.
    """
    f0 = fcsts[0]
    fv_units = f0[f_var][2]['units']
    t = f0['time']
    lat = f0['latitude'][1][0]
    lon = f0['longitude'][1][0]
    f_times = xray.decode_cf_datetime(t[1], t[2]['units'])
    f_times = f_times.astype('M8[h]')

    data = [f[f_var][1].ravel() for f in fcsts]
    df = pd.DataFrame(data, columns=f_times)
    dcount = df.apply(pd.value_counts).fillna(0)
    new_index = np.array(dcount.index).astype('float64').round(1)
    dcount.index = new_index
    xlim = (0, np.ceil(dcount.max().max()))

    plot_rows = int(np.ceil(len(f_times) / float(plot_cols)))
    fig, axes =  plt.subplots(plot_rows, plot_cols, sharex=True, sharey=True)
    i, j = (0, 0)
    for count, t in enumerate(f_times):
        i = count / plot_cols
        j = count % plot_cols
        dcount[t].plot(kind='barh', ax=axes[i, j], xlim=xlim, title="%s"
                % t, color='k', alpha=0.7)

    plt.show()
    plt.close()

def setup_parser(p, script):
    variable_choices=['wind_speed', 'pressure']
    p.add_argument(
            '--input', metavar='FILE', required='True',
            type=argparse.FileType('rb'), help="input file with windbreaker "
            "SPOT forecast ensemble")
    p.add_argument(
            '--variable', metavar='VARIABLE', required='True',
            choices=variable_choices, help="forecast variable for which to "
            "create plot; valid choices: %s" % ', '.join(variable_choices))
    p.add_argument(
            '--plotcol', metavar='NUMBER', type=int, default=4,
            help="number of columns for plot grid")


if __name__ == '__main__':

    script = os.path.basename(__file__)
    parser = argparse.ArgumentParser(description="""
    %s - shows distribution of forecast variable values across an ensemble of
    SPOT forecasts.""" % script)
    setup_parser(parser, script)
    args = parser.parse_args()

    payload = args.input.read()
    args.input.close()
    fcsts = [base64.b64decode(x) for x in zlib.decompress(payload).split('\t')]
    fcsts = [tinylib.beaufort_to_dict(f) for f in fcsts]

    display_espot(fcsts, args.variable, args.plotcol)
