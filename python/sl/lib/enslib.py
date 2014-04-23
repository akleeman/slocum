#!/usr/bin/python2.7

import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xray
from sl.lib import units
from sl.lib import conventions as conv

_fcst_vars = [(conv.WIND_SPEED, 'knot'),
             (conv.PRESSURE, 'hPa')]

def plot_spot_ensemble(fcsts, f_var=None, plot_type='box', save_path='None'):
    """
    Plots the spread of values for forecast variable f_var in an ensemble
    forecast along the forecast times

    Parameters:
    -----------
    fcsts: sequence
        List dictionaries with members of the ensemble.
    plot_type: str
        Either 'bar' or 'box'.
    f_var: str
        Forecast variable to plot. If no variable is specified, a combined
        box plot for pressure and wind will be generated.
    save_path: str
        If specified, the plot will be saved in directory save_path under the
        file name 'se_<lat,lon>_<t0>_<plot_type>.svg'. If not specified, the
        plot will displayed interactively.
    """
    plot_handler = {
            'bar': _plot_bar,
            'box': _plot_box
            }

    n = len(fcsts)
    f0 = fcsts[0]
    t = f0[conv.TIME]
    t0_stamp = np.datetime64(
            t[2][conv.UNITS].replace('hours since ', '') + 'Z')
    lat = f0[conv.LAT][1][0]
    lon = f0[conv.LON][1][0]
    f_times = xray.decode_cf_datetime(t[1], t[2][conv.UNITS])
    f_times = f_times.astype('M8[h]')

    title = ("SPOT Ensemble for lat: %.1f lon: %.1f\n($n = %d$, $t_0 =$ %s)" %
            (lat, lon, n, t0_stamp.item().strftime('%Y-%m-%dT%HZ')))

    # not pretty but works for now...
    data_list = []
    for var, plot_units in _fcst_vars:
        fv_units = f0[var][2][conv.UNITS]
        data = [units.convert_array(f[var][1].ravel(), fv_units,
                plot_units) for f in fcsts]
        if var == f_var:
            plot_handler[plot_type](
                    f_var, data, plot_units, f_times, title)
            break
        else:
            data_list.append(data)
    # print combined plot with data_list if no forecast variable
    # was specified:
    if not f_var:
        fig, axes = plt.subplots(2, 1, sharex=True)
        fig.suptitle(title)
        for i, (var, plot_units) in enumerate(_fcst_vars):
            ax = axes[i]
            ax.boxplot(np.array(data_list[i]))
            x_labels = [ft.item().strftime('%d/%HZ') for ft in f_times]
            ax.set_ylabel("%s [%s]" % (var, plot_units))
            ax.grid(axis='y')
        ax.set_xticklabels(x_labels, rotation='vertical')

    if save_path:
        lat_name = 'S' if lat < 0 else 'N'
        lon_name = 'W' if lon < 0 else 'E'
        time_str = t0_stamp.item().strftime('%Y%m%dT%HZ')
        file_name = 'se_%s_%4.1f%s-%5.1f%s' % (time_str,
                abs(lat), lat_name, abs(lon), lon_name)
        if f_var:
            file_name += '_%s' % f_var
        if plot_type:
            file_name += '_%s' % plot_type
        file_name += '.svg'
        plt.savefig(os.path.join( save_path, file_name), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def _plot_bar(f_var, data, plot_units, f_times, title):

    plot_cols = int(np.ceil(np.sqrt(len(f_times))))

    df = pd.DataFrame(data, columns=f_times)

    dcount = df.apply(pd.value_counts).fillna(0)
    new_index = np.array(dcount.index).astype('f8').round().astype('i4')
    dcount.index = new_index
    xlim = (0, np.ceil(dcount.max().max()))

    plot_rows = int(np.ceil(len(f_times) / float(plot_cols)))
    fig, axes =  plt.subplots(plot_rows, plot_cols, sharex=True, sharey=True)
    fig.suptitle("%s: %s" % (f_var, title))

    i, j = (0, 0)
    for count, t in enumerate(f_times):
        i = count / plot_cols
        j = count % plot_cols
        p_title = t.item().strftime('%d/%HZ')
        dcount[t].plot(kind='barh', ax=axes[i, j], xlim=xlim,
                color='k', alpha=0.7, fontsize='small')
        axes[i, j].set_title(p_title, fontsize='small')
        if j == 0:
            axes[i, j].set_yticklabels(dcount.index,fontsize='small')
            axes[i, j].set_ylabel("%s" % plot_units)
        if i == plot_rows - 1:
            axes[i, j].set_xlabel("count")
        plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)


def _plot_box(f_var, data, plot_units, f_times, title):

    fig, ax = plt.subplots()
    ax.boxplot(np.array(data))
    x_labels = [ft.item().strftime('%d/%HZ') for ft in f_times]
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.set_ylabel("%s [%s]" % (f_var, plot_units))
    ax.grid(axis='y')
    ax.set_title(title)

