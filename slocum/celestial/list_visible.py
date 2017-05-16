import ephem
import argparse
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from slocum.celestial import utils


def compute(time, lon, lat, body):
    obs = ephem.Observer()
    obs.lon = str(lon)
    obs.lat = str(lat)
    dt = pd.to_datetime(time)
    obs.date = dt.strftime('%Y/%m/%d %H:%M:%S')
    body.compute(obs)
    return body


def altitude_azimuth_to_xy(alt, az):
  y = (90 - alt) * np.cos(np.deg2rad(az))
  x = (90 - alt) * np.sin(np.deg2rad(az))
  return x, y


def plot_bodies(bodies):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    circle = plt.Circle((0, 0), 90., fill=False)
    ax.add_patch(circle)

    for rad in range(0, 90, 10):
        circle = plt.Circle((0, 0), rad, fill=False, color='black', alpha=0.2)
        ax.add_patch(circle)

    for body_name, body in bodies:
        alt = np.rad2deg(body.alt)
        az = np.rad2deg(body.az)
        if alt > 0. and body.mag < 3.:
          x, y = altitude_azimuth_to_xy(alt, az)
          radius = np.rad2deg(body.radius)

          if radius > 0.01:
            ax.add_patch(plt.Circle((x, y), 10 * radius, fill=True, color='yellow'))
            plt.text(x - len(body.name), y + 10 * radius + 2, body.name)
          elif isinstance(body, ephem.Planet):
            ax.add_patch(plt.Circle((x, y), 1, fill=True, color='white'))
            plt.text(x - len(body.name), y + 10 * radius + 2, body.name)
          else:
            plt.scatter(x, y, s=30 * np.exp(-body.mag), color='white')
            plt.text(x - len(body.name), y + 2, body.name)

    ax.grid(False)
    ax.set_axis_off()
    plt.text(0, 91, "N", fontsize=24, horizontalalignment='center', verticalalignment='bottom')
    plt.text(0, -91, "S", fontsize=24, horizontalalignment='center', verticalalignment='top')
    plt.text(91, 0, "E", fontsize=24, horizontalalignment='left', verticalalignment='center')
    plt.text(-91, 0, "W", fontsize=24, horizontalalignment='right', verticalalignment='center')
    plt.text(x, y + 2, body.name,
             horizontalalignment='center',
             verticalalignment='bottom')
    fig.set_tight_layout(True)

def main(args):
    if args.time is None:
        args.time = np.datetime64(datetime.datetime.utcnow())
    
    bodies = sorted(utils.get_bodies().iteritems(),
                    key=lambda x: x[0])

    bodies = [(body_name, compute(args.time, args.longitude, args.latitude, body))
              for body_name, body in bodies]

    for body_name, body in bodies:
      alt = np.rad2deg(body.alt)
      az = np.rad2deg(body.az)
      if alt > 0.:
        print ("%17s(%4.1f) would have elevation %5.2f and azimuth %6.2f at %s" %
               (body_name, body.mag, alt, az, args.time))

    plot_bodies(bodies)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    tracker.py
  
    A tool to determine your latitude from a series of sights and DR.""")
    parser.add_argument('latitude',
                        help=("The latitude of the observer."),
                        type=float)
    parser.add_argument('longitude',
                        help=("The longitude of the observer."),
                        type=float)
    parser.add_argument('--time',
                        help=("The time of the observation (UTC)."),
                        default=None,
                        type=np.datetime64)
    args = parser.parse_args()
    main(args)
