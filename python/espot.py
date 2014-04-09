#!/usr/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xray
from sl.lib import units
from sl.lib import conventions as conv

display_units = {
        conv.WIND_SPEED: 'knot',
        conv.PRESSURE: 'hPa'
        }

def display_espot(fcsts, f_var, plot_type='box'):
    """
    Plots the spread of values for forecast variable f_var in an ensemble
    forecast along the forecast times contained in fcsts, a list of
    dictionaries with members of the ensemble. plot_type is either 'bar' or
    'box'.
    """
    plot_handler = {
            'bar': plot_bar,
            'box': plot_box
            }

    f0 = fcsts[0]
    fv_units = f0[f_var][2][conv.UNITS]
    t = f0[conv.TIME]
    t0_stamp = np.datetime64(
            t[2][conv.UNITS].replace('hours since ', '') + 'Z')
    lat = f0[conv.LAT][1][0]
    lon = f0[conv.LON][1][0]
    f_times = xray.decode_cf_datetime(t[1], t[2][conv.UNITS])
    f_times = f_times.astype('M8[h]')

    title = ("SPOT Ensemble for lat: %.1f lon: %.1f\n($t_0$ = %s)" %
            (lat, lon, t0_stamp.item().strftime('%Y-%m-%dT%HZ')))

    data = [units.convert_array(f[f_var][1].ravel(), fv_units,
            display_units[f_var]) for f in fcsts]

    plot_handler[plot_type](f_var, data, f_times, title)


def plot_bar(f_var, data, f_times, title):

    plot_cols = 4

    df = pd.DataFrame(data, columns=f_times)

    dcount = df.apply(pd.value_counts).fillna(0)
    new_index = np.array(dcount.index).astype('f8').round().astype('i4')
    dcount.index = new_index
    xlim = (0, np.ceil(dcount.max().max()))

    plot_rows = int(np.ceil(len(f_times) / float(plot_cols)))
    fig, axes =  plt.subplots(plot_rows, plot_cols, sharex=True, sharey=True)
    fig.suptitle(title)

    i, j = (0, 0)
    for count, t in enumerate(f_times):
        i = count / plot_cols
        j = count % plot_cols
        dcount[t].plot(kind='barh', ax=axes[i, j], xlim=xlim, title="%s"
                % t, color='k', alpha=0.7)
        if j == 0:
            axes[i, j].set_ylabel("%s [%s]" % (f_var, display_units[f_var]))
        if i == plot_rows - 1:
            axes[i, j].set_xlabel("ensemble members [count]")

    plt.show()
    plt.close()

def plot_box(f_var, data, f_times, title):

    fig, ax = plt.subplots()
    ax.boxplot(np.array(data))
    x_labels = [ft.item().strftime('%d/%HZ') for ft in f_times]
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.set_ylabel("%s [%s]" % (f_var, display_units[f_var]))
    ax.grid(axis='y')
    ax.set_title(title)

    plt.show()
    plt.close()


if __name__ == '__main__':

    import sys
    import os.path
    import zlib
    import base64
    import argparse
    from sl.lib import tinylib

    def setup_parser(p, script):
        variable_choices=[conv.WIND_SPEED, conv.PRESSURE]
        p.add_argument(
                '--input', metavar='FILE', required='True',
                type=argparse.FileType('rb'), help="input file with "
                "windbreaker SPOT forecast ensemble")
        p.add_argument(
                '--b64', action='store_true', help="if specified, input file "
                "is expected to base64 encoded (as extracted from Sailmail "
                "email)")
        p.add_argument(
                '--variable', metavar='VARIABLE', required='True',
                choices=variable_choices, help="forecast variable for which "
                "to create plot; valid choices: %s" %
                ', '.join(variable_choices))
        p.add_argument(
                '--plot', metavar='TYPE', choices=['box', 'bar'],
                default='box', help="plot type to be created "
                "('bar'|'box'), defaults to 'box'")

    script = os.path.basename(__file__)
    parser = argparse.ArgumentParser(description="""
    %s - shows distribution of forecast variable values across an ensemble of
    SPOT forecasts.""" % script)
    setup_parser(parser, script)
    args = parser.parse_args()

    payload = args.input.read()
    args.input.close()
    if args.b64:
        payload = base64.b64decode(payload)
    fcsts = [base64.b64decode(x) for x in zlib.decompress(payload).split('\t')]
    fcsts = [tinylib.beaufort_to_dict(f) for f in fcsts]

    display_espot(fcsts, args.variable, args.plot)
